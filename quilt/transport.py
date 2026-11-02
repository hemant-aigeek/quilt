"""Peer links: authenticated, multiplexed request/response over TCP.

A ``Link`` wraps one TCP connection. Either side can issue requests; each
request carries an ``rid`` and the peer answers with a frame carrying the
same ``rid``. Unsolicited frames (no ``rid``, or a type the handler treats as
one-way) are delivered to the registered handler.

Membership of the mesh is gated by a shared secret: the dialer proves it
knows the secret by HMAC-ing a nonce the listener chose, and vice versa.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import itertools
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from . import protocol as P

log = logging.getLogger("quilt.transport")

Handler = Callable[["Link", dict], Awaitable[dict | None]]


class LinkClosed(Exception):
    pass


class AuthError(Exception):
    pass


def _mac(secret: bytes, nonce: bytes, node_id: str) -> bytes:
    return hmac.new(secret, nonce + node_id.encode(), hashlib.sha256).digest()


class Link:
    _ids = itertools.count(1)

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, handler: Handler):
        self.reader = reader
        self.writer = writer
        self.handler = handler
        self.peer_id: str | None = None
        self.peer_hello: dict = {}
        self._pending: dict[int, asyncio.Future] = {}
        self._closed = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._wlock = asyncio.Lock()

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self._task = asyncio.create_task(self._pump(), name=f"link-{self.peer_id}")

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self.writer.close()
            await asyncio.wait_for(self.writer.wait_closed(), 1.0)
        except Exception:
            pass
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(LinkClosed(f"link to {self.peer_id} closed"))
        self._pending.clear()
        if self._task and self._task is not asyncio.current_task():
            self._task.cancel()

    # -- sending -------------------------------------------------------------
    async def send(self, msg: dict) -> None:
        if self._closed.is_set():
            raise LinkClosed(f"link to {self.peer_id} closed")
        data = P.pack(msg)
        async with self._wlock:
            self.writer.write(data)
            try:
                await self.writer.drain()
            except (ConnectionError, asyncio.CancelledError, OSError) as e:
                await self.close()
                raise LinkClosed(str(e)) from e

    async def request(self, msg: dict, timeout: float | None = 30.0) -> dict:
        rid = next(self._ids)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            await self.send({**msg, "rid": rid})
            resp = await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(rid, None)
        if resp.get("t") == P.ERROR:
            raise RuntimeError(f"{self.peer_id}: {resp.get('error')}")
        return resp

    # -- receiving -----------------------------------------------------------
    async def _pump(self) -> None:
        try:
            while not self._closed.is_set():
                frame = await P.read_frame(self.reader)
                if frame is None:
                    break
                rid = frame.get("rid")
                if rid is not None and rid in self._pending and "t" in frame and frame.get("reply"):
                    fut = self._pending.get(rid)
                    if fut and not fut.done():
                        fut.set_result(frame)
                    continue
                asyncio.create_task(self._dispatch(frame))
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("link %s pump failed", self.peer_id)
        finally:
            await self.close()

    async def _dispatch(self, frame: dict) -> None:
        rid = frame.get("rid")
        try:
            resp = await self.handler(self, frame)
        except Exception as e:  # noqa: BLE001 - surface to the caller
            log.debug("handler error for %s: %s", frame.get("t"), e)
            resp = {"t": P.ERROR, "error": f"{type(e).__name__}: {e}"}
        if rid is not None and resp is not None:
            try:
                await self.send({**resp, "rid": rid, "reply": True})
            except LinkClosed:
                pass


# --- handshake ----------------------------------------------------------------

async def _exchange_hello(link: Link, secret: bytes, my_hello, *, dialer: bool) -> None:
    """Mutual HMAC challenge. Both sides send a nonce, both prove the secret."""
    if callable(my_hello):
        my_hello = my_hello()
    nonce = os.urandom(16)
    await link.send({"t": P.HELLO, "v": P.PROTOCOL_VERSION, "nonce": nonce, **my_hello})
    frame = await asyncio.wait_for(P.read_frame(link.reader), 10.0)
    if not frame or frame.get("t") != P.HELLO:
        raise AuthError("expected hello")
    if frame.get("v") != P.PROTOCOL_VERSION:
        raise AuthError(f"protocol version mismatch: {frame.get('v')}")
    their_nonce = frame["nonce"]
    await link.send({"t": P.WELCOME, "mac": _mac(secret, their_nonce, my_hello["node_id"])})
    frame2 = await asyncio.wait_for(P.read_frame(link.reader), 10.0)
    if not frame2 or frame2.get("t") != P.WELCOME:
        raise AuthError("expected welcome")
    expect = _mac(secret, nonce, frame["node_id"])
    if not hmac.compare_digest(expect, frame2.get("mac", b"")):
        raise AuthError("peer failed mesh-secret challenge")
    link.peer_id = frame["node_id"]
    link.peer_hello = {k: v for k, v in frame.items() if k not in ("t", "v", "nonce")}


async def dial(addr: P.Addr, secret: bytes, my_hello, handler: Handler, timeout: float = 5.0) -> Link:
    reader, writer = await asyncio.wait_for(asyncio.open_connection(addr.host, addr.port), timeout)
    link = Link(reader, writer, handler)
    try:
        await _exchange_hello(link, secret, my_hello, dialer=True)
    except Exception:
        await link.close()
        raise
    link.start()
    return link


async def listen(
    bind: P.Addr,
    secret: bytes,
    my_hello,
    handler: Handler,
    on_link: Callable[[Link], Awaitable[None]],
) -> asyncio.base_events.Server:
    async def accept(reader, writer):
        link = Link(reader, writer, handler)
        try:
            await _exchange_hello(link, secret, my_hello, dialer=False)
        except Exception as e:  # noqa: BLE001
            log.warning("rejected inbound connection: %s", e)
            await link.close()
            return
        link.start()
        await on_link(link)

    return await asyncio.start_server(accept, bind.host, bind.port)


def hello_for(node_id: str, listen_addr: P.Addr, extra: dict[str, Any] | None = None) -> dict:
    return {"node_id": node_id, "addr": str(listen_addr), **(extra or {})}
