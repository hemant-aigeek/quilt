import json

import aiohttp
from conftest import start_mesh, wait_ready

from quilt.api import serve


async def test_http_chat_and_status():
    nodes = await start_mesh(2)
    runners = []
    try:
        leader = await wait_ready(nodes)
        follower = next(n for n in nodes if n is not leader)
        ports = {}
        for n in nodes:
            r = await serve(n, "127.0.0.1", 0)
            runners.append(r)
            ports[n.node_id] = r.addresses[0][1]
            n.membership.api_url = f"http://127.0.0.1:{ports[n.node_id]}"
        async with aiohttp.ClientSession() as http:
            async with http.get(f"http://127.0.0.1:{ports[leader.node_id]}/v1/mesh") as r:
                s = await r.json()
                assert s["is_leader"] and s["plan"]["version"] >= 1
            async with http.post(f"http://127.0.0.1:{ports[leader.node_id]}/v1/chat",
                                 json={"session": "h", "prompt": "hi", "max_tokens": 5, "temperature": 0}) as r:
                assert r.status == 200
                events = [json.loads(l) for l in (await r.text()).splitlines() if l]
            assert sum("token" in e for e in events) == 5 and events[-1]["done"]
            # the follower does not generate; it points at the leader
            async with http.post(f"http://127.0.0.1:{ports[follower.node_id]}/v1/chat", json={"prompt": "x"},
                                 allow_redirects=False) as r:
                assert r.status == 307
            async with http.get(f"http://127.0.0.1:{ports[follower.node_id]}/v1/sessions/h") as r:
                assert [t["role"] for t in (await r.json())["turns"]] == ["user", "assistant"]
            async with http.get(f"http://127.0.0.1:{ports[follower.node_id]}/") as r:
                assert "Quilt" in await r.text()
    finally:
        for r in runners:
            await r.cleanup()
        for n in nodes:
            await n.stop()
