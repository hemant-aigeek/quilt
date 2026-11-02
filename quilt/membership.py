"""Membership: who is in the quilt, how to reach them, and whether they are alive.

Every member keeps one link to every other member (a full mesh — the quilt is
meant for a handful of devices, not thousands). Members learn about each
other by gossip: each hello carries the sender's known peer addresses, and
unknown addresses are dialled. Liveness is a round-trip ping; a member that
misses ``dead_after`` seconds of pongs, or whose link drops, is marked gone.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from . import protocol as P
from .profile import DeviceProfile
from .transport import Link, dial, hello_for, listen

log = logging.getLogger("quilt.membership")


@dataclass
class Member:
    node_id: str
    addr: P.Addr
    profile: DeviceProfile | None = None
    api_url: str | None = None
    link: Link | None = None
    last_pong: float = field(default_factory=time.monotonic)
    joined_at: float = field(default_factory=time.monotonic)

    @property
    def alive(self) -> bool:
        return self.link is not None and not self.link.closed


class Membership:
    def __init__(
        self,
        node_id: str,
        listen_addr: P.Addr,
        secret: bytes,
        profile: DeviceProfile,
        handler: Callable[[Link, dict], Awaitable[dict | None]],
        *,
        api_url: str | None = None,
        advertise: P.Addr | None = None,
        ping_interval: float = 1.0,
        dead_after: float = 3.0,
    ):
        self.node_id = node_id
        self._bind = listen_addr
        self.listen_addr = advertise or listen_addr  # what peers are told to dial
        self.secret = secret
        self.profile = profile
        self.api_url = api_url
        self._handler = handler
        self.ping_interval = ping_interval
        self.dead_after = dead_after
        self.members: dict[str, Member] = {}
        self._server = None
        self._tasks: list[asyncio.Task] = []
        self._dialing: set[str] = set()
        self.on_change: list[Callable[[], None]] = []  # fired on join/leave
        self.ping_payload: Callable[[], dict] = dict  # extra fields piggybacked on every ping
        self.on_pong: Callable[[str, dict], None] = lambda node_id, pong: None
        self._stopped = False

    # -- lifecycle -----------------------------------------------------------
    def _hello(self) -> dict:
        return hello_for(
            self.node_id,
            self.listen_addr,
            {
                "profile": self.profile.to_dict(),
                "api_url": self.api_url,
                "peers": {m.node_id: str(m.addr) for m in self.members.values()},
            },
        )

    async def start(self, bootstrap: list[P.Addr]) -> None:
        self._server = await listen(self._bind, self.secret, self._hello, self._handler, self._on_inbound)
        port = self._server.sockets[0].getsockname()[1]
        self.listen_addr = P.Addr(self.listen_addr.host, port)
        for addr in bootstrap:
            self._tasks.append(asyncio.create_task(self._dial_loop(addr)))
        self._tasks.append(asyncio.create_task(self._ping_loop()))

    async def stop(self) -> None:
        self._stopped = True
        for t in self._tasks:
            t.cancel()
        for m in list(self.members.values()):
            if m.link:
                await m.link.close()
        if self._server:
            self._server.close()
            try:  # 3.12+ waits for every accepted connection; ours are closed above
                await asyncio.wait_for(self._server.wait_closed(), 2.0)
            except asyncio.TimeoutError:
                pass

    # -- link management -----------------------------------------------------
    async def _on_inbound(self, link: Link) -> None:
        self._adopt(link, dialer=link.peer_id)

    def _adopt(self, link: Link, *, dialer: str) -> None:
        """Register a freshly authenticated link, resolving simultaneous dials."""
        pid = link.peer_id
        assert pid
        if pid == self.node_id:
            asyncio.create_task(link.close())
            return
        hello = link.peer_hello
        member = self.members.get(pid)
        if member is None:
            member = Member(pid, P.Addr.parse(hello["addr"]))
            self.members[pid] = member
        member.profile = DeviceProfile.from_dict(hello["profile"]) if hello.get("profile") else member.profile
        member.api_url = hello.get("api_url")
        if member.alive and member.link is not link:
            # Both sides dialled at once. Keep the link whose dialer has the lower id.
            existing_dialer = getattr(member.link, "_dialer", None)
            if existing_dialer is not None and existing_dialer < dialer:
                asyncio.create_task(link.close())
                return
            asyncio.create_task(member.link.close())
        link._dialer = dialer  # type: ignore[attr-defined]
        member.link = link
        member.last_pong = time.monotonic()
        log.info("%s: member %s joined (%s)", self.node_id, pid, member.addr)
        asyncio.create_task(self._watch(member, link))
        for addr in hello.get("peers", {}).values():
            self.ensure_peer(P.Addr.parse(addr))
        asyncio.create_task(self._gossip())
        self._fire()

    async def _gossip(self) -> None:
        """Tell every member about every member, so late joiners get introduced."""
        peers = {m.node_id: str(m.addr) for m in self.members.values() if m.alive}
        peers[self.node_id] = str(self.listen_addr)
        for m in list(self.members.values()):
            if m.alive:
                try:
                    await m.link.send({"t": P.GOSSIP, "peers": peers})
                except Exception:  # noqa: BLE001
                    pass

    def on_gossip(self, m: dict) -> None:
        for node_id, addr in m.get("peers", {}).items():
            if node_id != self.node_id:
                self.ensure_peer(P.Addr.parse(addr))

    async def _watch(self, member: Member, link: Link) -> None:
        await link.wait_closed()
        if member.link is link:
            log.info("%s: member %s left", self.node_id, member.node_id)
            member.link = None
            self._fire()
            if not self._stopped:
                self._tasks.append(asyncio.create_task(self._dial_loop(member.addr)))

    def ensure_peer(self, addr: P.Addr) -> None:
        if addr == self.listen_addr or self._stopped:
            return
        if any(m.addr == addr and m.alive for m in self.members.values()):
            return
        self._tasks.append(asyncio.create_task(self._dial_loop(addr)))

    async def _dial_loop(self, addr: P.Addr) -> None:
        key = str(addr)
        if key in self._dialing:
            return
        self._dialing.add(key)
        delay = 0.2
        try:
            while not self._stopped:
                if any(m.addr == addr and m.alive for m in self.members.values()):
                    return
                try:
                    link = await dial(addr, self.secret, self._hello, self._handler)
                except Exception as e:  # noqa: BLE001
                    log.debug("%s: dial %s failed: %s", self.node_id, addr, e)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 5.0)
                    continue
                self._adopt(link, dialer=self.node_id)
                return
        finally:
            self._dialing.discard(key)

    # -- liveness ------------------------------------------------------------
    async def _ping_loop(self) -> None:
        while not self._stopped:
            await asyncio.sleep(self.ping_interval)
            now = time.monotonic()
            for m in list(self.members.values()):
                if not m.alive:
                    continue
                if now - m.last_pong > self.dead_after:
                    log.warning("%s: member %s unresponsive; dropping link", self.node_id, m.node_id)
                    await m.link.close()
                    continue
                asyncio.create_task(self._ping(m))

    async def _ping(self, m: Member) -> None:
        try:
            pong = await m.link.request({"t": P.PING, **self.ping_payload()}, timeout=self.dead_after)
            m.last_pong = time.monotonic()
            self.on_pong(m.node_id, pong)
        except Exception:
            pass

    # -- queries -------------------------------------------------------------
    def alive_ids(self) -> set[str]:
        return {self.node_id} | {m.node_id for m in self.members.values() if m.alive}

    def profiles(self) -> list[DeviceProfile]:
        out = [self.profile]
        out += [m.profile for m in self.members.values() if m.alive and m.profile]
        return out

    def link_to(self, node_id: str) -> Link:
        m = self.members.get(node_id)
        if m is None or not m.alive:
            raise ConnectionError(f"no live link to {node_id}")
        return m.link

    def subscribe(self, cb: Callable[[], None]) -> None:
        self.on_change.append(cb)

    def _fire(self) -> None:
        for cb in list(self.on_change):
            try:
                cb()
            except Exception:  # noqa: BLE001
                log.exception("membership callback failed")

    def note_profile(self, node_id: str, profile: dict) -> None:
        m = self.members.get(node_id)
        if m:
            m.profile = DeviceProfile.from_dict(profile)

    def snapshot(self) -> list[dict]:
        rows = [{"node_id": self.node_id, "addr": str(self.listen_addr), "alive": True, "self": True,
                 "profile": self.profile.to_dict(), "api_url": self.api_url}]
        for m in self.members.values():
            rows.append({"node_id": m.node_id, "addr": str(m.addr), "alive": m.alive, "self": False,
                         "profile": m.profile.to_dict() if m.profile else None, "api_url": m.api_url})
        return rows
