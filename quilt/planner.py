"""The planner: cut a model into patches and pin each patch to a member.

Goal: minimise the slowest patch (the pipeline's bottleneck) subject to each
member's memory. The search is over a *time budget* ``T``: given ``T``, a
member with layer rate ``r`` may take at most ``floor(T * r)`` layers, and at
most as many as fit in its memory. Members are visited in a fixed order and
each takes as many layers as it may; if the layers run out, ``T`` is feasible.
The smallest feasible ``T`` is found by bisecting the finite set of candidate
budgets ``{k / r_i}``.

Every patch also gets a *standby*: a different member that keeps a warm copy of
the patch's weights. When the owner disappears, the standby takes over without
a full re-plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .profile import DeviceProfile


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_layers: int
    hidden: int
    heads: int
    vocab: int
    max_seq: int
    bytes_per_layer: int
    embed_bytes: int
    head_bytes: int
    kv_bytes_per_layer: int = 0  # reserved per layer for the KV cache at max_seq

    @property
    def layer_footprint(self) -> int:
        return self.bytes_per_layer + self.kv_bytes_per_layer

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> ModelSpec:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Patch:
    owner: str
    start: int
    stop: int
    standby: str | None = None

    @property
    def layers(self) -> range:
        return range(self.start, self.stop)

    def __len__(self) -> int:
        return self.stop - self.start

    def to_dict(self) -> dict:
        return {"owner": self.owner, "start": self.start, "stop": self.stop, "standby": self.standby}

    @classmethod
    def from_dict(cls, d: dict) -> Patch:
        return cls(d["owner"], d["start"], d["stop"], d.get("standby"))


@dataclass
class Plan:
    version: int
    model: str
    patches: list[Patch] = field(default_factory=list)
    bottleneck_s: float = 0.0  # estimated time of the slowest patch per token

    @property
    def members(self) -> list[str]:
        return [p.owner for p in self.patches]

    def owners_and_standbys(self, node_id: str) -> tuple[list[int], list[int]]:
        own = [i for i, p in enumerate(self.patches) if p.owner == node_id]
        sb = [i for i, p in enumerate(self.patches) if p.standby == node_id]
        return own, sb

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model": self.model,
            "patches": [p.to_dict() for p in self.patches],
            "bottleneck_s": self.bottleneck_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Plan:
        return cls(d["version"], d["model"], [Patch.from_dict(p) for p in d["patches"]], d.get("bottleneck_s", 0.0))


class PlanError(Exception):
    pass


def _capacity(p: DeviceProfile, spec: ModelSpec, *, first: bool, last: bool) -> int:
    overhead = (spec.embed_bytes if first else 0) + (spec.head_bytes if last else 0)
    return max(0, (p.mem_bytes - overhead) // spec.layer_footprint)


def _order(profiles: list[DeviceProfile], preferred_first: str | None) -> list[DeviceProfile]:
    ordered = sorted(profiles, key=lambda p: (-p.layer_rate, p.node_id))
    if preferred_first:
        ordered.sort(key=lambda p: p.node_id != preferred_first)
    return ordered


def _assign(ordered: list[DeviceProfile], spec: ModelSpec, budget: float) -> list[Patch] | None:
    patches: list[Patch] = []
    cursor = 0
    for p in ordered:
        if cursor >= spec.n_layers:
            break
        remaining = spec.n_layers - cursor
        by_time = math.floor(budget * p.layer_rate + 1e-9)
        # Last-patch overhead applies if this member would finish the model.
        cap_mid = _capacity(p, spec, first=(cursor == 0), last=False)
        cap_end = _capacity(p, spec, first=(cursor == 0), last=True)
        take = min(by_time, cap_mid, remaining)
        if take >= remaining:
            take = min(by_time, cap_end, remaining)
            if take < remaining:
                # Cannot host the head too; take a middle slice and let someone else finish.
                take = min(by_time, cap_mid, remaining - 1)
        if take <= 0:
            continue
        patches.append(Patch(p.node_id, cursor, cursor + take))
        cursor += take
    return patches if cursor >= spec.n_layers else None


def plan(profiles: list[DeviceProfile], spec: ModelSpec, *, version: int, preferred_first: str | None = None,
         min_members: int = 1) -> Plan:
    if not profiles:
        raise PlanError("no members to plan over")
    ordered = _order(profiles, preferred_first)
    candidates = sorted({k / p.layer_rate for p in ordered for k in range(1, spec.n_layers + 1)})
    lo, hi = 0, len(candidates) - 1
    best: list[Patch] | None = None
    if _assign(ordered, spec, candidates[hi]) is None:
        total_cap = sum(_capacity(p, spec, first=False, last=False) for p in ordered)
        raise PlanError(
            f"model {spec.name} needs {spec.n_layers} layers but members can hold {total_cap} in total"
        )
    while lo <= hi:
        mid = (lo + hi) // 2
        got = _assign(ordered, spec, candidates[mid])
        if got is None:
            lo = mid + 1
        else:
            best, hi = got, mid - 1
    assert best is not None
    if len(best) < min_members:
        raise PlanError(f"plan uses {len(best)} member(s); {min_members} required")
    _assign_standbys(best, profiles, spec)
    bottleneck = max(len(pt) / next(p.layer_rate for p in profiles if p.node_id == pt.owner) for pt in best)
    return Plan(version=version, model=spec.name, patches=best, bottleneck_s=bottleneck)


def _assign_standbys(patches: list[Patch], profiles: list[DeviceProfile], spec: ModelSpec) -> None:
    """Pick, for each patch, a member with enough spare memory to warm a copy."""
    by_id = {p.node_id: p for p in profiles}
    spare = {p.node_id: p.mem_bytes for p in profiles}
    for i, pt in enumerate(patches):
        extra = (spec.embed_bytes if i == 0 else 0) + (spec.head_bytes if i == len(patches) - 1 else 0)
        spare[pt.owner] -= len(pt) * spec.layer_footprint + extra
    for i, pt in enumerate(patches):
        need = len(pt) * spec.layer_footprint
        need += (spec.embed_bytes if i == 0 else 0) + (spec.head_bytes if i == len(patches) - 1 else 0)
        choices = [n for n, s in spare.items() if n != pt.owner and s >= need]
        if not choices:
            continue
        # Prefer the member with the most spare memory, then the fastest.
        pick = max(choices, key=lambda n: (spare[n], by_id[n].layer_rate))
        pt.standby = pick
        spare[pick] -= need


def plan_after_loss(current: Plan, lost: set[str], profiles: list[DeviceProfile], spec: ModelSpec,
                    *, version: int, preferred_first: str | None = None) -> Plan:
    """Fast path: hand each orphaned patch to its standby; otherwise re-plan."""
    alive = {p.node_id for p in profiles} - lost
    patches = [Patch(p.owner, p.start, p.stop, p.standby) for p in current.patches]
    for pt in patches:
        if pt.owner in alive:
            continue
        if pt.standby in alive:
            pt.owner, pt.standby = pt.standby, None
        else:
            return plan([p for p in profiles if p.node_id in alive], spec, version=version,
                        preferred_first=preferred_first)
    patches = _merge_adjacent(patches)
    for pt in patches:  # standbys that vanished are simply cleared
        pt.standby = None
    _assign_standbys(patches, [p for p in profiles if p.node_id in alive], spec)
    rates = {p.node_id: p.layer_rate for p in profiles}
    bottleneck = max(len(pt) / rates[pt.owner] for pt in patches)
    return Plan(version=version, model=current.model, patches=patches, bottleneck_s=bottleneck)


def _merge_adjacent(patches: list[Patch]) -> list[Patch]:
    """Neighbouring patches with one owner become a single patch (one fewer hop)."""
    merged: list[Patch] = []
    for pt in patches:
        if merged and merged[-1].owner == pt.owner and merged[-1].stop == pt.start:
            merged[-1] = Patch(pt.owner, merged[-1].start, pt.stop)
        else:
            merged.append(Patch(pt.owner, pt.start, pt.stop, pt.standby))
    return merged


def validate(plan_: Plan, spec: ModelSpec) -> None:
    cursor = 0
    for pt in plan_.patches:
        if pt.start != cursor or pt.stop <= pt.start:
            raise PlanError(f"patches are not contiguous at layer {cursor}: {pt}")
        cursor = pt.stop
    if cursor != spec.n_layers:
        raise PlanError(f"plan covers {cursor} of {spec.n_layers} layers")
