"""Multi-member behaviour on loopback: planning, generation, replication, failover."""


import pytest
from conftest import run_turn, start_mesh, wait_for, wait_ready


async def stop_all(nodes):
    for n in nodes:
        await n.stop()


async def test_single_member_leads_itself():
    (n,) = await start_mesh(1)
    try:
        leader = await wait_ready([n])
        assert leader is n and n.term == 1
        assert [(p.start, p.stop) for p in n.plan.patches] == [(0, 6)]
        text, final = await run_turn(n, "s", "hi", max_new_tokens=4)
        assert final["tokens"] == 4 and len(text) >= 1
    finally:
        await stop_all([n])


async def test_three_members_share_layers_and_replicate(mesh3):
    leader = await wait_ready(mesh3)
    assert len(leader.plan.patches) == 3
    assert sorted(leader.plan.members) == ["n0", "n1", "n2"]
    text, final = await run_turn(leader, "s", "hello", max_new_tokens=6)
    assert final["tokens"] == 6
    await wait_for(lambda: all(n.journal.seq == 2 for n in mesh3), what="journal replication")
    for n in mesh3:
        assert [e.role for e in n.journal.session("s")] == ["user", "assistant"]
    # a second turn reuses the pipeline caches: only the new tokens are fed
    await run_turn(leader, "s", "again", max_new_tokens=3)
    # everything except the last sampled token and the turn terminator has been through the pipeline
    assert leader._fed["s"] == len(leader.journal.session_tokens("s")) - 2


async def test_pipeline_output_matches_single_machine():
    """The distributed answer must equal what one machine computes alone (greedy)."""
    nodes = await start_mesh(3)
    solo = await start_mesh(1)
    try:
        leader = await wait_ready(nodes)
        alone = await wait_ready(solo)
        a, _ = await run_turn(leader, "s", "deterministic", max_new_tokens=10)
        b, _ = await run_turn(alone, "s", "deterministic", max_new_tokens=10)
        assert a == b
    finally:
        await stop_all(nodes + solo)


async def test_leader_loss_is_survived():
    nodes = await start_mesh(3)
    try:
        leader = await wait_ready(nodes)
        await run_turn(leader, "s", "first", max_new_tokens=4)
        await wait_for(lambda: all(n.journal.seq == 2 for n in nodes), what="replication")
        await leader.stop()
        rest = [n for n in nodes if n is not leader]
        new = await wait_ready(rest, timeout=15)
        assert new.term == 2 and new.stats["failovers"] == 1
        assert new.plan.version >= 2 and leader.node_id not in new.plan.members
        # the session continues from the replicated journal
        text, final = await run_turn(new, "s", "second", max_new_tokens=4)
        assert final["tokens"] == 4
        assert [e.text for e in new.journal.session("s") if e.role == "user"] == ["first", "second"]
        await wait_for(lambda: all(n.journal.seq == 4 for n in rest), what="post-failover replication")
    finally:
        await stop_all(nodes)


async def test_worker_loss_uses_standby():
    nodes = await start_mesh(3)
    try:
        leader = await wait_ready(nodes)
        victim = next(n for n in nodes if n is not leader and n.node_id in leader.plan.members)
        lost_patch = next(p for p in leader.plan.patches if p.owner == victim.node_id)
        await victim.stop()
        await wait_for(lambda: leader.plan_ready.is_set() and victim.node_id not in leader.plan.members, 15,
                       "re-plan after worker loss")
        covering = next(p for p in leader.plan.patches if p.start <= lost_patch.start < p.stop)
        assert covering.owner == lost_patch.standby
        text, final = await run_turn(leader, "s", "still", max_new_tokens=4)
        assert final["tokens"] == 4
    finally:
        await stop_all(nodes)


async def test_late_joiner_receives_journal_and_is_planned_in():
    nodes = await start_mesh(2)
    try:
        leader = await wait_ready(nodes)
        await run_turn(leader, "s", "before", max_new_tokens=3)
        late = (await start_mesh(1))[0]
        # join the existing mesh by dialling the leader
        late.membership.ensure_peer(leader.listen_addr)
        nodes.append(late)
        await wait_for(lambda: late.journal.seq == 2, 10, "journal snapshot for late joiner")
        await wait_for(lambda: leader.plan_ready.is_set() and late.node_id in leader.plan.members, 10, "re-plan")
        assert late.leader_id == leader.node_id
    finally:
        await stop_all(nodes)


async def test_wrong_secret_is_rejected():
    (n,) = await start_mesh(1)
    try:
        from quilt.transport import AuthError, dial

        with pytest.raises(AuthError):
            await dial(n.listen_addr, b"wrong", {"node_id": "x", "addr": "127.0.0.1:1"}, lambda l, m: None)
    finally:
        await stop_all([n])


async def test_generate_on_follower_raises_not_leader(mesh3):
    leader = await wait_ready(mesh3)
    follower = next(n for n in mesh3 if n is not leader)
    from quilt.node import NotLeader

    with pytest.raises(NotLeader) as ei:
        async for _ in follower.generate("s", "x"):
            pass
    assert ei.value.leader_id == leader.node_id
