"""HTTP face of a member: a streaming chat endpoint, mesh status, and a small
dashboard. Any member serves it; a member that is not the leader answers chat
requests with a redirect to the leader's address, so a client (or the
dashboard) follows the leadership wherever it moves.
"""

from __future__ import annotations

import json
from importlib import resources

from aiohttp import web

from .node import NotLeader, QuiltNode

NODE = web.AppKey("node", QuiltNode)


@web.middleware
async def _cors(req: web.Request, handler):
    """The dashboard may be served by one member and talk to the leader on another."""
    if req.method == "OPTIONS":
        resp = web.Response()
    else:
        try:
            resp = await handler(req)
        except web.HTTPException as e:
            resp = e
    if resp.prepared:  # a streaming body already sent its headers
        return resp
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    if isinstance(resp, web.HTTPException):
        raise resp
    return resp


def build_app(node: QuiltNode) -> web.Application:
    app = web.Application(client_max_size=4 * 1024 * 1024, middlewares=[_cors])
    app[NODE] = node
    app.add_routes([
        web.get("/", index),
        web.get("/healthz", healthz),
        web.get("/v1/mesh", mesh),
        web.get("/v1/sessions/{session}", session),
        web.delete("/v1/sessions/{session}", forget),
        web.post("/v1/chat", chat),
    ])
    return app


async def index(req: web.Request) -> web.Response:
    html = resources.files("quilt").joinpath("dashboard.html").read_text()
    return web.Response(text=html, content_type="text/html")


async def healthz(req: web.Request) -> web.Response:
    node: QuiltNode = req.app[NODE]
    return web.json_response({"ok": True, "node_id": node.node_id, "leader": node.leader_id})


async def mesh(req: web.Request) -> web.Response:
    node: QuiltNode = req.app[NODE]
    return web.json_response(node.status())


async def session(req: web.Request) -> web.Response:
    node: QuiltNode = req.app[NODE]
    sid = req.match_info["session"]
    return web.json_response({"session": sid, "turns": [e.to_dict() for e in node.journal.session(sid)]})


def _redirect(node: QuiltNode, exc: NotLeader, path: str) -> web.Response:
    if exc.leader_api:
        raise web.HTTPTemporaryRedirect(exc.leader_api.rstrip("/") + path)
    raise web.HTTPServiceUnavailable(text=json.dumps({"error": "no leader yet", "leader": exc.leader_id}),
                                     content_type="application/json")


async def forget(req: web.Request) -> web.Response:
    node: QuiltNode = req.app[NODE]
    try:
        await node.forget_session(req.match_info["session"])
    except NotLeader as e:
        return _redirect(node, e, req.path)
    return web.json_response({"ok": True})


async def chat(req: web.Request) -> web.StreamResponse:
    """Body: {"session": str, "prompt": str, "max_tokens": int, "temperature": float}.

    Streams newline-delimited JSON: ``{"token": "..."}`` events then one
    ``{"done": true, ...}`` event (or ``{"error": "..."}``).
    """
    node: QuiltNode = req.app[NODE]
    body = await req.json()
    if not node.is_leader:
        m = node.membership.members.get(node.leader_id or "")
        return _redirect(node, NotLeader(node.leader_id, m.api_url if m else None), req.path)
    resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson", "Cache-Control": "no-cache",
                                       "Access-Control-Allow-Origin": "*"})  # set before the stream is prepared
    await resp.prepare(req)
    try:
        async for ev in node.generate(
            str(body.get("session", "default")), str(body["prompt"]),
            max_new_tokens=int(body.get("max_tokens", 128)), temperature=float(body.get("temperature", 0.8)),
            top_k=int(body.get("top_k", 40)), stop_on_newline=bool(body.get("stop_on_newline", False)),
        ):
            await resp.write((json.dumps(ev) + "\n").encode())
    except NotLeader as e:
        await resp.write((json.dumps({"error": "leadership moved", "leader": e.leader_id, "leader_api": e.leader_api}) + "\n").encode())
    except Exception as e:  # noqa: BLE001
        await resp.write((json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n").encode())
    await resp.write_eof()
    return resp


async def serve(node: QuiltNode, host: str, port: int) -> web.AppRunner:
    runner = web.AppRunner(build_app(node), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    return runner
