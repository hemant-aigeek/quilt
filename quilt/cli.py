"""Command line: ``quilt join`` starts a member, ``quilt chat`` talks to one,
``quilt plan`` previews a layout, ``quilt bench`` measures this machine."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import socket
import sys
from pathlib import Path

from . import __version__
from . import protocol as P


def _secret(args) -> bytes:
    s = args.secret or os.environ.get("QUILT_SECRET")
    if not s:
        sys.exit("a mesh secret is required: --secret ... or QUILT_SECRET=...")
    return s.encode()


def _backend(args):
    if args.model == "toy":
        from .model.toy import ToyBackend, ToyConfig

        return ToyBackend(ToyConfig(hidden=args.toy_hidden, heads=args.toy_heads, layers=args.toy_layers))
    from .model.hf import HFBackend, HFConfig

    return HFBackend(HFConfig(model_id=args.model, dtype=args.dtype, max_seq=args.max_seq))


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="toy", help="'toy' (built-in, no download) or a Hugging Face model id")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-seq", type=int, default=4096)
    p.add_argument("--toy-layers", type=int, default=8)
    p.add_argument("--toy-hidden", type=int, default=64)
    p.add_argument("--toy-heads", type=int, default=4)


def _guess_host() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


async def _join(args) -> None:
    from .api import serve
    from .node import QuiltNode

    backend = _backend(args)
    listen = P.Addr.parse(args.listen)
    advertise_host = args.advertise or (listen.host if listen.host not in ("0.0.0.0", "::") else _guess_host())
    api_host, api_port = P.Addr.parse(args.api).host, P.Addr.parse(args.api).port
    node = QuiltNode(
        args.id or f"{socket.gethostname()}-{secrets.token_hex(2)}",
        listen, _secret(args), backend, [P.Addr.parse(a) for a in args.join],
        api_url=f"http://{advertise_host}:{api_port}", advertise=P.Addr(advertise_host, listen.port),
        mem_bytes=int(args.memory_gib * (1 << 30)) if args.memory_gib else None, layer_rate=args.layer_rate,
        device=args.device, journal_path=Path(args.journal) if args.journal else None,
        min_members=args.min_members,
    )
    await node.start()
    runner = await serve(node, api_host, api_port)
    print(f"quilt {__version__} · member {node.node_id} · mesh {node.listen_addr} · dashboard http://{advertise_host}:{api_port}")
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        await node.stop()


async def _chat(args) -> None:
    import aiohttp

    session = args.session or f"cli-{secrets.token_hex(3)}"
    url = args.url.rstrip("/") + "/v1/chat"
    async with aiohttp.ClientSession() as http:
        while True:
            try:
                prompt = args.prompt or input("you> ")
            except EOFError:
                return
            if not prompt.strip():
                continue
            body = {"session": session, "prompt": prompt, "max_tokens": args.max_tokens, "temperature": args.temperature}
            async with http.post(url, json=body, allow_redirects=True) as r:
                if r.status != 200:
                    print(f"[{r.status}] {await r.text()}")
                    if args.prompt:
                        sys.exit(1)
                    continue
                print("quilt> ", end="", flush=True)
                async for raw in r.content:
                    line = raw.decode().strip()
                    if not line:
                        continue
                    ev = json.loads(line)
                    if "token" in ev:
                        print(ev["token"], end="", flush=True)
                    elif ev.get("done"):
                        print(f"\n   [{ev['tokens']} tokens · {ev['seconds']}s · plan v{ev['plan_version']} · term {ev['term']}]")
                    elif "error" in ev:
                        print(f"\n   [error: {ev['error']}]")
            if args.prompt:
                return


def _plan(args) -> None:
    from .planner import plan
    from .profile import DeviceProfile

    backend = _backend(args)
    rows = json.loads(Path(args.profiles).read_text())
    profiles = [DeviceProfile.from_dict(r) for r in rows]
    p = plan(profiles, backend.spec, version=1, preferred_first=args.first)
    print(json.dumps(p.to_dict(), indent=2))


def _bench(args) -> None:
    from .profile import build_profile

    backend = _backend(args)
    prof = build_profile(args.id or socket.gethostname(), mem_bytes=None, hidden=backend.spec.hidden,
                         heads=backend.spec.heads, device=args.device)
    print(json.dumps(prof.to_dict(), indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="quilt", description="stitch spare machines into one language model")
    ap.add_argument("--version", action="version", version=f"quilt {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    j = sub.add_parser("join", help="start a member (the first member simply joins nobody)")
    j.add_argument("--id", help="member name (default: hostname + random suffix)")
    j.add_argument("--listen", default="0.0.0.0:7700", help="mesh bind address")
    j.add_argument("--advertise", help="host peers should dial (default: detected LAN address)")
    j.add_argument("--api", default="0.0.0.0:7780", help="HTTP bind address for the dashboard and chat API")
    j.add_argument("--join", action="append", default=[], metavar="HOST:PORT", help="existing member(s) to join")
    j.add_argument("--secret", help="mesh secret shared by all members (or QUILT_SECRET)")
    j.add_argument("--memory-gib", type=float, help="memory to lend (default: 60%% of what is free)")
    j.add_argument("--layer-rate", type=float, help="skip the start-up benchmark and declare layers/second")
    j.add_argument("--device", default="cpu", help="torch device, e.g. cpu, cuda, mps")
    j.add_argument("--journal", help="file to persist the conversation journal")
    j.add_argument("--min-members", type=int, default=1)
    _model_args(j)
    j.set_defaults(fn=lambda a: asyncio.run(_join(a)))

    c = sub.add_parser("chat", help="chat with a quilt through any member's API")
    c.add_argument("--url", default="http://127.0.0.1:7780")
    c.add_argument("--session")
    c.add_argument("--prompt", help="one-shot prompt instead of a REPL")
    c.add_argument("--max-tokens", type=int, default=128)
    c.add_argument("--temperature", type=float, default=0.8)
    c.set_defaults(fn=lambda a: asyncio.run(_chat(a)))

    pl = sub.add_parser("plan", help="preview the layout for a set of device profiles (JSON list)")
    pl.add_argument("profiles")
    pl.add_argument("--first", help="member that should host the first patch")
    _model_args(pl)
    pl.set_defaults(fn=_plan)

    b = sub.add_parser("bench", help="measure what this machine would contribute")
    b.add_argument("--id")
    b.add_argument("--device", default="cpu")
    _model_args(b)
    b.set_defaults(fn=_bench)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
