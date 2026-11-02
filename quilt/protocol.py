"""Wire protocol: length-prefixed msgpack frames with an inline tensor codec.

Every frame is a msgpack map with at least a ``t`` (type) key. Request frames
carry an ``rid`` (request id) which the response echoes back, so a single
persistent link can multiplex many in-flight calls in both directions.

Tensors travel as a small map ``{"__t": 1, "dtype": ..., "shape": [...],
"data": <bin>}``. Only the data buffer is copied; no pickling, no JSON.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 512 * 1024 * 1024
_HEADER = struct.Struct("!I")

# --- message types ----------------------------------------------------------
HELLO = "hello"
WELCOME = "welcome"
PING = "ping"
PONG = "pong"
GOSSIP = "gossip"  # peer list / profiles
PLAN = "plan"  # leader -> all: adopt this layer plan
PLAN_ACK = "plan_ack"
FORWARD = "forward"  # coordinator -> first patch, patch -> next patch
LOGITS = "logits"  # last patch -> coordinator
DROP_SESSION = "drop_session"
JOURNAL_APPEND = "journal_append"
JOURNAL_ACK = "journal_ack"
JOURNAL_SNAPSHOT = "journal_snapshot"
LEAD = "lead"  # leader heartbeat with term + plan version
STEP_DOWN = "step_down"
ERROR = "error"
OK = "ok"


class ProtocolError(Exception):
    pass


def encode_tensor(x: Any) -> dict:
    """Encode a torch tensor or numpy array as a msgpack-friendly map."""
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().contiguous()
        if str(x.dtype) == "torch.bfloat16":  # numpy has no bf16; ship as fp16
            x = x.half()
        arr = x.numpy()
    else:
        arr = np.ascontiguousarray(x)
    return {"__t": 1, "dtype": arr.dtype.str, "shape": list(arr.shape), "data": arr.tobytes()}


def decode_tensor(m: dict, *, torch_out: bool = True) -> Any:
    arr = np.frombuffer(m["data"], dtype=np.dtype(m["dtype"])).reshape(m["shape"])
    if not torch_out:
        return arr
    import torch

    return torch.from_numpy(arr.copy())


def _default(obj: Any) -> Any:
    if hasattr(obj, "detach") or isinstance(obj, np.ndarray):
        return encode_tensor(obj)
    if isinstance(obj, range):
        return [obj.start, obj.stop]
    raise TypeError(f"cannot encode {type(obj)!r}")


def pack(msg: dict) -> bytes:
    body = msgpack.packb(msg, default=_default, use_bin_type=True)
    if len(body) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame too large: {len(body)} bytes")
    return _HEADER.pack(len(body)) + body


def unpack(body: bytes) -> dict:
    return msgpack.unpackb(body, raw=False)


async def read_frame(reader) -> dict | None:
    """Read one frame; return None on clean EOF."""
    hdr = await reader.readexactly(_HEADER.size) if not reader.at_eof() else b""
    if not hdr:
        return None
    (n,) = _HEADER.unpack(hdr)
    if n > MAX_FRAME_BYTES:
        raise ProtocolError(f"incoming frame too large: {n}")
    body = await reader.readexactly(n)
    return unpack(body)


def is_tensor(v: Any) -> bool:
    return isinstance(v, dict) and v.get("__t") == 1


@dataclass(frozen=True)
class Addr:
    host: str
    port: int

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"

    @classmethod
    def parse(cls, s: str) -> Addr:
        host, _, port = s.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"bad address {s!r}; expected host:port")
        return cls(host, int(port))

