"""Device profiles: what a member can contribute to the quilt.

A profile is deliberately small — bytes of memory the member will lend and a
*layer rate* (layers it can run per second for one token). The rate is
measured, not declared: at start-up a member times a synthetic transformer
block of the model's width on its own hardware.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import asdict, dataclass, field


@dataclass
class DeviceProfile:
    node_id: str
    mem_bytes: int  # memory this member lends to model weights + caches
    layer_rate: float  # layers / second for a single-token step (measured)
    device: str = "cpu"
    host: str = field(default_factory=platform.node)
    cores: int = field(default_factory=lambda: os.cpu_count() or 1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DeviceProfile:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def available_memory_bytes() -> int:
    """Best-effort free memory. Falls back to 2 GiB when unknown."""
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(pages * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 2 << 30


def measure_layer_rate(hidden: int, heads: int, device: str = "cpu", reps: int = 8) -> float:
    """Time one decoder block on a single token with a short synthetic context.

    Returns layers/second. The exact number matters less than its ratio across
    members; the planner only uses it to balance stage times.
    """
    import torch

    from .model.toy import ToyBlock, ToyConfig

    cfg = ToyConfig(hidden=hidden, heads=heads, layers=1, vocab=256, max_seq=512)
    block = ToyBlock(cfg).to(device).eval()
    x = torch.randn(1, 1, hidden, device=device)
    with torch.inference_mode():
        block(x, 0, None)  # warm-up
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        best = float("inf")
        for _ in range(reps):  # best-of-N: robust to other processes starting at the same time
            t0 = time.perf_counter()
            block(x, 0, None)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
    return 1.0 / max(best, 1e-6)


def build_profile(node_id: str, *, mem_bytes: int | None, hidden: int, heads: int, device: str = "cpu",
                  layer_rate: float | None = None) -> DeviceProfile:
    mem = mem_bytes if mem_bytes is not None else int(available_memory_bytes() * 0.6)
    rate = layer_rate if layer_rate is not None else measure_layer_rate(hidden, heads, device)
    return DeviceProfile(node_id=node_id, mem_bytes=mem, layer_rate=rate, device=device)
