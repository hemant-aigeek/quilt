from __future__ import annotations

import asyncio
import logging

import pytest

from quilt import protocol as P
from quilt.model.toy import ToyBackend, ToyConfig
from quilt.node import QuiltNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

SECRET = b"test-quilt-secret"


def toy_backend(layers: int = 6) -> ToyBackend:
    return ToyBackend(ToyConfig(hidden=32, heads=4, layers=layers, vocab=256, max_seq=256))


async def start_mesh(n: int, *, layers: int = 6, mem_mib: list[float] | None = None, rates: list[float] | None = None,
                     lead_timeout: float = 0.6, min_members: int = 1) -> list[QuiltNode]:
    """Start ``n`` members on loopback; the first is the bootstrap peer."""
    nodes: list[QuiltNode] = []
    spec = toy_backend(layers).spec
    for i in range(n):
        mem = int((mem_mib[i] if mem_mib else 64) * (1 << 20))
        rate = rates[i] if rates else 1000.0
        bootstrap = [nodes[0].listen_addr] if nodes else []
        node = QuiltNode(
            f"n{i}", P.Addr("127.0.0.1", 0), SECRET, toy_backend(layers), bootstrap,
            api_url=f"http://127.0.0.1:{9000 + i}", mem_bytes=mem, layer_rate=rate,
            lead_interval=0.15, lead_timeout=lead_timeout, ping_interval=0.2, dead_after=0.8,
            min_members=min_members,
        )
        await node.start()
        nodes.append(node)
    assert spec.n_layers == layers
    return nodes


async def wait_for(pred, timeout: float = 10.0, what: str = "condition"):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if pred():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


async def wait_ready(nodes: list[QuiltNode], timeout: float = 10.0) -> QuiltNode:
    def ready():
        leaders = [n for n in nodes if n.is_leader and n.plan_ready.is_set()]
        return len(leaders) == 1 and all(n.leader_id == leaders[0].node_id for n in nodes)

    await wait_for(ready, timeout, "a single ready leader")
    return next(n for n in nodes if n.is_leader)


async def run_turn(node: QuiltNode, session: str, prompt: str, max_new_tokens: int = 8) -> tuple[str, dict]:
    pieces, final = [], {}
    async for ev in node.generate(session, prompt, max_new_tokens=max_new_tokens, temperature=0.0):
        if "token" in ev:
            pieces.append(ev["token"])
        else:
            final = ev
    return "".join(pieces), final


@pytest.fixture
async def mesh3():
    nodes = await start_mesh(3)
    try:
        yield nodes
    finally:
        for n in nodes:
            await n.stop()
