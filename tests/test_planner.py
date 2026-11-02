import pytest

from quilt.planner import ModelSpec, Patch, Plan, PlanError, plan, plan_after_loss, validate
from quilt.profile import DeviceProfile

SPEC = ModelSpec(name="m", n_layers=12, hidden=64, heads=4, vocab=256, max_seq=256,
                 bytes_per_layer=100, embed_bytes=50, head_bytes=50, kv_bytes_per_layer=0)


def prof(node_id, mem, rate):
    return DeviceProfile(node_id=node_id, mem_bytes=mem, layer_rate=rate)


def test_balances_by_speed():
    p = plan([prof("fast", 10_000, 200.0), prof("slow", 10_000, 100.0)], SPEC, version=1)
    validate(p, SPEC)
    by = {pt.owner: len(pt) for pt in p.patches}
    assert by == {"fast": 8, "slow": 4}
    assert p.bottleneck_s == pytest.approx(0.04)


def test_memory_limits_layers():
    # "big" is fast but can only hold 4 layers (+ embed overhead); the rest spills to "small"
    p = plan([prof("big", 450, 1000.0), prof("small", 10_000, 10.0)], SPEC, version=1)
    validate(p, SPEC)
    by = {pt.owner: len(pt) for pt in p.patches}
    assert by["big"] == 4 and by["small"] == 8


def test_preferred_first_hosts_embedding():
    p = plan([prof("a", 10_000, 100.0), prof("b", 10_000, 100.0)], SPEC, version=1, preferred_first="b")
    assert p.patches[0].owner == "b" and p.patches[0].start == 0


def test_not_enough_memory_is_an_error():
    with pytest.raises(PlanError):
        plan([prof("tiny", 300, 100.0)], SPEC, version=1)


def test_standbys_are_other_members_with_room():
    p = plan([prof("a", 10_000, 100.0), prof("b", 10_000, 100.0), prof("c", 10_000, 100.0)], SPEC, version=1)
    for pt in p.patches:
        assert pt.standby is not None and pt.standby != pt.owner


def test_min_members():
    with pytest.raises(PlanError):
        plan([prof("a", 10_000, 100.0)], SPEC, version=1, min_members=2)


def test_loss_hands_patch_to_standby_and_merges():
    profiles = [prof("a", 10_000, 100.0), prof("b", 10_000, 100.0), prof("c", 10_000, 100.0)]
    current = Plan(1, "m", [Patch("a", 0, 4, "b"), Patch("b", 4, 8, "c"), Patch("c", 8, 12, "a")])
    new = plan_after_loss(current, {"a"}, profiles, SPEC, version=2)
    validate(new, SPEC)
    assert [(pt.owner, pt.start, pt.stop) for pt in new.patches] == [("b", 0, 8), ("c", 8, 12)]
    assert all(pt.standby in {"b", "c"} and pt.standby != pt.owner for pt in new.patches)


def test_loss_without_standby_replans():
    profiles = [prof("a", 10_000, 100.0), prof("b", 10_000, 100.0)]
    current = Plan(1, "m", [Patch("a", 0, 6, None), Patch("b", 6, 12, None)])
    new = plan_after_loss(current, {"a"}, profiles, SPEC, version=2)
    validate(new, SPEC)
    assert new.members == ["b"]


def test_plan_roundtrip():
    p = plan([prof("a", 10_000, 100.0), prof("b", 10_000, 50.0)], SPEC, version=3)
    assert Plan.from_dict(p.to_dict()) == p
