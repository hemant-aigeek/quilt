"""A quilt member: hosts patches, replicates the journal, and — when elected —
leads: plans the layer layout and drives generation through the mesh.

Leadership is a *lease*. The leader broadcasts ``LEAD`` every
``lead_interval`` seconds with its term, plan version and journal sequence.
A member that hears nothing for ``lead_timeout`` runs *deterministic
succession*: among the members it can still reach, the one with the most
complete journal wins, ties broken by measured speed and then node id. There
is no voting round — every survivor computes the same answer from the same
gossip, and the winner simply starts leading with a higher term. A stale
leader that reappears hears the higher term and steps down.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path

import torch

from . import protocol as P
from .journal import Entry, Journal
from .membership import Membership
from .model.base import ModelBackend, PatchRuntime
from .planner import ModelSpec, Plan, PlanError, plan_after_loss, validate
from .planner import plan as make_plan
from .profile import build_profile
from .transport import Link

log = logging.getLogger("quilt.node")

USER_PREFIX = "User: "
ASSISTANT_PREFIX = "\nAssistant: "
TURN_END = "\n"


class NotLeader(Exception):
    def __init__(self, leader_id: str | None, leader_api: str | None):
        super().__init__(f"not the leader (leader={leader_id})")
        self.leader_id = leader_id
        self.leader_api = leader_api


class QuiltNode:
    def __init__(
        self,
        node_id: str,
        listen: P.Addr,
        secret: bytes,
        backend: ModelBackend,
        bootstrap: list[P.Addr] | None = None,
        *,
        api_url: str | None = None,
        advertise: P.Addr | None = None,
        mem_bytes: int | None = None,
        layer_rate: float | None = None,
        device: str = "cpu",
        journal_path: Path | None = None,
        lead_interval: float = 0.5,
        lead_timeout: float = 2.0,
        ping_interval: float = 1.0,
        dead_after: float = 3.0,
        min_members: int = 1,
    ):
        self.node_id = node_id
        self.backend = backend
        self.spec: ModelSpec = backend.spec
        self.device = device
        self.bootstrap = bootstrap or []
        self.min_members = min_members
        self.lead_interval = lead_interval
        self.lead_timeout = lead_timeout
        self.journal = Journal(journal_path)
        profile = build_profile(node_id, mem_bytes=mem_bytes, hidden=self.spec.hidden, heads=self.spec.heads,
                                device=device, layer_rate=layer_rate)
        self.membership = Membership(node_id, listen, secret, profile, self._handle, api_url=api_url,
                                     advertise=advertise, ping_interval=ping_interval, dead_after=dead_after)
        # leadership state
        self.term = 0
        self.leader_id: str | None = None
        self._last_lead = time.monotonic()
        self._started_at = time.monotonic()
        self._suspects: set[str] = set()
        self._peer_seq: dict[str, int] = {}
        self._peer_term: dict[str, int] = {}
        # plan state
        self.plan: Plan | None = None
        self.plan_ready = asyncio.Event()  # leader: plan acked by every member it uses
        self._patches: dict[int, PatchRuntime] = {}
        self._runtimes: dict[tuple, PatchRuntime] = {}  # (start, stop, embed, head) -> runtime
        self._replan_task: asyncio.Task | None = None
        self._plan_lock = asyncio.Lock()
        self._fed: dict[str, int] = {}  # leader-only: tokens cached in the pipeline per session
        self._snapshot_inflight = False
        self._tasks: list[asyncio.Task] = []
        self._stopped = False
        self.events: collections.deque[str] = collections.deque(maxlen=200)
        self.stats = {"tokens": 0, "turns": 0, "failovers": 0, "started": time.time()}

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.membership.ping_payload = lambda: {"seq": self.journal.seq, "term": self.term}
        self.membership.on_pong = self._on_pong
        await self.membership.start(self.bootstrap)
        self.membership.subscribe(self._on_membership)
        self._tasks.append(asyncio.create_task(self._lead_loop(), name=f"lead-{self.node_id}"))
        self._note(f"started at {self.membership.listen_addr}; lending {self.membership.profile.mem_bytes >> 20} MiB, "
                   f"{self.membership.profile.layer_rate:.0f} layers/s")

    async def stop(self) -> None:
        self._stopped = True
        self.leader_id = None
        self.plan_ready.clear()
        for t in self._tasks:
            t.cancel()
        if self._replan_task:
            self._replan_task.cancel()
        await self.membership.stop()
        for rt in self._runtimes.values():
            rt.unload()

    @property
    def is_leader(self) -> bool:
        return self.leader_id == self.node_id

    @property
    def listen_addr(self) -> P.Addr:
        return self.membership.listen_addr

    def _note(self, msg: str) -> None:
        log.info("%s: %s", self.node_id, msg)
        self.events.append(f"{time.strftime('%H:%M:%S')} {msg}")

    # ------------------------------------------------------------------ messages
    def _on_pong(self, node_id: str, pong: dict) -> None:
        self._peer_seq[node_id] = pong.get("seq", 0)
        self._peer_term[node_id] = pong.get("term", 0)

    async def _handle(self, link: Link, m: dict) -> dict | None:
        t = m.get("t")
        if t == P.PING:
            self._on_pong(link.peer_id, m)
            return {"t": P.PONG, "seq": self.journal.seq, "term": self.term}
        if t == P.LEAD:
            await self._on_lead(link.peer_id, m)
            return None
        if t == P.GOSSIP:
            self.membership.on_gossip(m)
            return None
        if t == P.PLAN:
            await self._apply_plan(Plan.from_dict(m["plan"]))
            return {"t": P.PLAN_ACK, "version": m["plan"]["version"]}
        if t == P.FORWARD:
            return await self._forward_hop(m)
        if t == P.DROP_SESSION:
            for rt in self._runtimes.values():
                rt.drop_session(m["session"])
            return {"t": P.OK}
        if t == P.JOURNAL_APPEND:
            ok = self.journal.accept(Entry.from_dict(m["entry"]))
            return {"t": P.JOURNAL_ACK, "seq": self.journal.seq, "need_snapshot": not ok}
        if t == P.JOURNAL_SNAPSHOT:
            if "entries" in m:  # pushed by the leader
                self._merge_entries([Entry.from_dict(e) for e in m["entries"]])
                return {"t": P.OK, "seq": self.journal.seq}
            return {"t": P.JOURNAL_SNAPSHOT, "entries": [e.to_dict() for e in self.journal.since(m.get("since", 0))]}
        if t == P.STEP_DOWN:
            return {"t": P.OK}
        return {"t": P.ERROR, "error": f"unknown message type {t!r}"}

    def _merge_entries(self, entries: list[Entry]) -> None:
        for e in entries:
            if not self.journal.accept(e):
                have = {x.seq: x for x in self.journal.entries}
                have.update({x.seq: x for x in entries})
                self.journal.replace(list(have.values()))
                return

    # ------------------------------------------------------------------ leadership
    async def _on_lead(self, sender: str, m: dict) -> None:
        term = m["term"]
        if term < self.term:
            return  # stale leader; it will hear our term in pings and step down
        if term == self.term and self.leader_id not in (None, sender):
            # Two leaders claim the same term (a partition healed, or a member elected itself
            # before it found the mesh). Everyone yields to the better-ranked claimant.
            if self._rank(self.leader_id) >= self._rank(sender):
                return
        if term > self.term or self.leader_id != sender:
            if self.is_leader and term > self.term:
                self._note(f"stepping down: {sender} leads with term {term}")
            elif self.leader_id != sender:
                self._note(f"following {sender} (term {term})")
            self.term = term
            self.leader_id = sender
            self._fed.clear()
        self._last_lead = time.monotonic()
        self._suspects.clear()
        if m.get("seq", 0) > self.journal.seq and not self._snapshot_inflight:
            asyncio.create_task(self._pull_snapshot(sender))

    async def _pull_snapshot(self, leader: str) -> None:
        self._snapshot_inflight = True
        try:
            link = self.membership.link_to(leader)
            resp = await link.request({"t": P.JOURNAL_SNAPSHOT, "since": self.journal.seq}, timeout=10)
            self._merge_entries([Entry.from_dict(e) for e in resp.get("entries", [])])
        except Exception as e:  # noqa: BLE001
            log.debug("%s: snapshot pull failed: %s", self.node_id, e)
        finally:
            self._snapshot_inflight = False

    async def _reconcile_journal(self) -> None:
        """A fresh leader adopts any entries a survivor holds that it does not."""
        results = await self._broadcast({"t": P.JOURNAL_SNAPSHOT, "since": self.journal.seq}, request=True, timeout=5)
        for r in results.values():
            if isinstance(r, dict) and r.get("entries"):
                self._merge_entries([Entry.from_dict(e) for e in r["entries"]])
                self._note(f"adopted {len(r['entries'])} journal entries from a survivor")

    async def _lead_loop(self) -> None:
        while not self._stopped:
            await asyncio.sleep(self.lead_interval)
            try:
                if self.is_leader:
                    await self._broadcast({"t": P.LEAD, "term": self.term, "seq": self.journal.seq,
                                           "plan": self.plan.version if self.plan else 0})
                elif time.monotonic() - self._last_lead > self.lead_timeout and self._may_elect():
                    self._elect()
            except Exception:  # noqa: BLE001
                log.exception("%s: lead loop error", self.node_id)

    def _may_elect(self) -> bool:
        """A member told to join peers waits for them before it would crown itself."""
        if not self.bootstrap or len(self.membership.alive_ids()) > 1:
            return True
        return time.monotonic() - self._started_at > 3 * self.lead_timeout

    def _rank(self, node_id: str) -> tuple:
        seq = self.journal.seq if node_id == self.node_id else self._peer_seq.get(node_id, -1)
        rate = (self.membership.profile.layer_rate if node_id == self.node_id
                else (self.membership.members[node_id].profile.layer_rate if self.membership.members[node_id].profile else 0))
        return (seq, rate, node_id)

    def _elect(self) -> None:
        alive = self.membership.alive_ids() - self._suspects
        if self.leader_id in alive and self.leader_id != self.node_id and self.leader_id in self.membership.members:
            # Lease expired but the link is up: the leader is overloaded or paused. Give it one more period.
            self._suspects.add(self.leader_id)
            self._last_lead = time.monotonic()
            return
        winner = max(alive, key=self._rank)
        log.debug("%s: election among %s -> %s (ranks %s)", self.node_id, sorted(alive), winner,
                  {n: self._rank(n) for n in alive})
        known_terms = [self.term, *self._peer_term.values()]
        if winner == self.node_id:
            old = self.leader_id
            self.term = max(known_terms) + 1
            self.leader_id = self.node_id
            self._last_lead = time.monotonic()
            self._fed.clear()
            if old is not None and old != self.node_id:
                self.stats["failovers"] += 1
                self._note(f"leader {old} lost; taking over (term {self.term}, journal seq {self.journal.seq})")
            else:
                self._note(f"leading (term {self.term})")
            self._schedule_replan(lost={old} if old else set())
            asyncio.create_task(self._reconcile_journal())
        else:
            # Expect the winner to announce itself within one lease; otherwise exclude it next time.
            self._suspects.add(winner)
            self._last_lead = time.monotonic()

    async def _broadcast(self, msg: dict, *, request: bool = False, timeout: float = 10.0) -> dict[str, dict | Exception]:
        results: dict[str, dict | Exception] = {}

        async def one(member):
            try:
                if request:
                    results[member.node_id] = await member.link.request(msg, timeout=timeout)
                else:
                    await member.link.send(msg)
            except Exception as e:  # noqa: BLE001
                results[member.node_id] = e

        await asyncio.gather(*(one(m) for m in self.membership.members.values() if m.alive))
        return results

    # ------------------------------------------------------------------ planning
    def _on_membership(self) -> None:
        if self._stopped:
            return
        alive = self.membership.alive_ids()
        if self.is_leader:
            lost = (set(self.plan.members) - alive) if self.plan else set()
            self._schedule_replan(lost=lost)
        elif self.leader_id and self.leader_id not in alive:
            self._last_lead = 0.0  # leader's link is gone: elect on the next tick

    def _schedule_replan(self, lost: set[str]) -> None:
        if self._replan_task and not self._replan_task.done():
            self._replan_task.cancel()
        self._replan_task = asyncio.create_task(self._replan(lost))

    async def _replan(self, lost: set[str]) -> None:
        await asyncio.sleep(0.3)  # debounce bursts of joins/leaves
        async with self._plan_lock:
            if not self.is_leader:
                return
            profiles = self.membership.profiles()
            version = (self.plan.version if self.plan else 0) + 1
            try:
                alive = {p.node_id for p in profiles}
                if self.plan and lost and alive == set(self.plan.members) - lost:
                    new = plan_after_loss(self.plan, lost, profiles, self.spec, version=version,
                                          preferred_first=self.node_id)
                    took_over = [f"{n.owner} took over layers {n.start}-{n.stop}" for o in self.plan.patches
                                 if o.owner in lost for n in new.patches if n.start <= o.start < n.stop]
                    if took_over:
                        self._note("standby " + "; ".join(took_over))
                else:
                    new = make_plan(profiles, self.spec, version=version, preferred_first=self.node_id,
                                    min_members=self.min_members)
                validate(new, self.spec)
            except PlanError as e:
                self.plan_ready.clear()
                self._note(f"cannot plan over {len(profiles)} member(s): {e}")
                return
            if self.plan and new.patches == self.plan.patches and self.plan_ready.is_set():
                return
            self.plan_ready.clear()
            self.plan = new
            layout = " | ".join(f"{p.owner}:{p.start}-{p.stop}" + (f"(sb {p.standby})" if p.standby else "")
                                for p in new.patches)
            self._note(f"plan v{new.version}: {layout}; ~{new.bottleneck_s * 1000:.1f} ms/token bottleneck")
            await self._apply_plan(new)
            acks = await self._broadcast({"t": P.PLAN, "plan": new.to_dict()}, request=True, timeout=120)
            missing = [n for n in set(new.members) - {self.node_id} if not isinstance(acks.get(n), dict)]
            if missing:
                self._note(f"plan v{new.version} not acknowledged by {missing}; re-planning")
                self._schedule_replan(lost=set(missing))
                return
            self._fed.clear()
            self.plan_ready.set()

    async def _apply_plan(self, p: Plan) -> None:
        """Load the patches this member owns or stands by for; drop stale caches."""
        self.plan = p
        own, standby = p.owners_and_standbys(self.node_id)
        wanted: dict[int, tuple] = {}
        for idx in own + standby:
            pt = p.patches[idx]
            wanted[idx] = (pt.start, pt.stop, idx == 0, idx == len(p.patches) - 1)
        new_patches: dict[int, PatchRuntime] = {}
        for idx, key in wanted.items():
            rt = self._runtimes.get(key)
            if rt is None:
                start, stop, emb, head = key
                rt = await asyncio.to_thread(self.backend.load_patch, range(start, stop), with_embed=emb,
                                             with_head=head, device=self.device)
                self._runtimes[key] = rt
            new_patches[idx] = rt
        for key in list(self._runtimes):
            if key not in wanted.values():
                self._runtimes.pop(key).unload()
        for rt in self._runtimes.values():  # positions are only meaningful within one plan
            rt.drop_all()
        self._patches = new_patches

    # ------------------------------------------------------------------ pipeline
    async def _forward_hop(self, m: dict) -> dict:
        if self.plan is None or m["plan_version"] != self.plan.version:
            raise RuntimeError(f"plan version mismatch (have {self.plan.version if self.plan else None})")
        idx = m["hop"]
        rt = self._patches.get(idx)
        if rt is None or self.plan.patches[idx].owner != self.node_id:
            raise RuntimeError(f"not the owner of patch {idx}")
        session, start = m["session"], m["start_pos"]

        def compute():
            if idx == 0:
                h = rt.embed(P.decode_tensor(m["tokens"]), start)
            else:
                h = P.decode_tensor(m["h"])
            h = rt.forward(session, h, start)
            if idx == len(self.plan.patches) - 1:
                return {"t": P.LOGITS, "logits": P.encode_tensor(rt.head(h[:, -1:])[0, -1].float())}
            return {"h": P.encode_tensor(h)}

        out = await asyncio.to_thread(compute)
        if "logits" in out:
            return out
        nxt = {**{k: v for k, v in m.items() if k not in ("tokens", "h", "rid", "reply")}, "hop": idx + 1, "h": out["h"]}
        owner = self.plan.patches[idx + 1].owner
        if owner == self.node_id:
            return await self._forward_hop(nxt)
        return await self.membership.link_to(owner).request(nxt, timeout=120)

    async def _step(self, session: str, tokens: list[int], start_pos: int) -> torch.Tensor:
        msg = {"t": P.FORWARD, "session": session, "plan_version": self.plan.version, "hop": 0,
               "start_pos": start_pos, "tokens": P.encode_tensor(torch.tensor([tokens], dtype=torch.int64))}
        owner = self.plan.patches[0].owner
        resp = await (self._forward_hop(msg) if owner == self.node_id
                      else self.membership.link_to(owner).request(msg, timeout=120))
        return P.decode_tensor(resp["logits"])

    # ------------------------------------------------------------------ generation
    def _require_leader(self) -> None:
        if not self.is_leader:
            m = self.membership.members.get(self.leader_id or "")
            raise NotLeader(self.leader_id, m.api_url if m else None)

    async def generate(self, session: str, prompt: str, *, max_new_tokens: int = 64, temperature: float = 0.8,
                       top_k: int = 40, stop_on_newline: bool = False) -> AsyncIterator[dict]:
        """Run one conversational turn. Yields ``{"token": str}`` per token, then ``{"done": ...}``."""
        self._require_leader()
        tok = self.backend.tokenizer
        user_tokens = tok.encode(USER_PREFIX + prompt + ASSISTANT_PREFIX)
        history = self.journal.session_tokens(session)
        context = history + user_tokens
        generated: list[int] = []
        t0 = time.monotonic()
        while len(generated) < max_new_tokens:
            self._require_leader()
            logits = await self._step_with_retry(session, context, generated)
            nxt = _sample(logits, temperature, top_k)
            generated.append(nxt)
            self.stats["tokens"] += 1
            piece = tok.decode([nxt])
            yield {"token": piece}
            if tok.eos_id is not None and nxt == tok.eos_id:
                break
            if stop_on_newline and "\n" in piece:
                break
        text = tok.decode(generated)
        self.stats["turns"] += 1
        await self._commit_turn(session, prompt, user_tokens, text, generated + tok.encode(TURN_END))
        yield {"done": True, "text": text, "tokens": len(generated), "seconds": round(time.monotonic() - t0, 3),
               "plan_version": self.plan.version if self.plan else None, "term": self.term}

    async def _step_with_retry(self, session: str, context: list[int], generated: list[int]) -> torch.Tensor:
        """Feed whatever the pipeline has not cached yet; replay from scratch after any disruption."""
        last_err: Exception | None = None
        for attempt in range(6):
            self._require_leader()
            try:
                await asyncio.wait_for(self.plan_ready.wait(), timeout=60)
                full = context + generated
                fed = self._fed.get(session, 0)
                if fed > len(full) - 1 or fed < 0:
                    fed = 0
                tokens = full[fed:]
                logits = await self._step(session, tokens, fed)
                self._fed[session] = len(full)
                return logits
            except (ConnectionError, RuntimeError, ValueError, asyncio.TimeoutError, OSError) as e:
                last_err = e
                self._fed.pop(session, None)
                self._note(f"step failed ({type(e).__name__}: {e}); retry {attempt + 1}")
                await asyncio.sleep(min(0.2 * (attempt + 1), 1.0))
        raise RuntimeError(f"generation failed after retries: {last_err}")

    async def _commit_turn(self, session: str, prompt: str, user_tokens: list[int], text: str,
                           reply_tokens: list[int]) -> None:
        for role, body, toks in (("user", prompt, user_tokens), ("assistant", text, reply_tokens)):
            entry = self.journal.append(session, role, body, toks, term=self.term)
            acks = await self._broadcast({"t": P.JOURNAL_APPEND, "entry": entry.to_dict()}, request=True, timeout=5)
            for node, r in acks.items():
                if isinstance(r, dict) and r.get("need_snapshot"):
                    try:
                        link = self.membership.link_to(node)
                        await link.request({"t": P.JOURNAL_SNAPSHOT,
                                            "entries": [e.to_dict() for e in self.journal.since(r.get("seq", 0))]})
                    except Exception:  # noqa: BLE001
                        pass

    async def forget_session(self, session: str) -> None:
        self._require_leader()
        self._fed.pop(session, None)
        for rt in self._runtimes.values():
            rt.drop_session(session)
        await self._broadcast({"t": P.DROP_SESSION, "session": session}, request=True, timeout=5)

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        leader = self.membership.members.get(self.leader_id or "")
        return {
            "node_id": self.node_id,
            "addr": str(self.listen_addr),
            "leader": self.leader_id,
            "is_leader": self.is_leader,
            "leader_api": self.membership.api_url if self.is_leader else (leader.api_url if leader else None),
            "term": self.term,
            "model": self.spec.to_dict(),
            "plan": self.plan.to_dict() if self.plan else None,
            "plan_ready": self.plan_ready.is_set() if self.is_leader else None,
            "members": self.membership.snapshot(),
            "journal_seq": self.journal.seq,
            "sessions": self.journal.sessions(),
            "stats": self.stats,
            "events": list(self.events)[-40:],
        }


def _sample(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1))
