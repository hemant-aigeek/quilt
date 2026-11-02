from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch

from ..planner import ModelSpec


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    @property
    def eos_id(self) -> int | None: ...


class PatchRuntime(ABC):
    """A loaded contiguous slice of layers, with per-session KV caches.

    Hidden states are ``[1, T, hidden]`` tensors. ``start_pos`` is the absolute
    position of the first token in ``h`` within the session, so the runtime can
    extend that session's cache rather than recompute it.
    """

    layers: range
    has_embed: bool
    has_head: bool

    @abstractmethod
    def embed(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor: ...

    @abstractmethod
    def forward(self, session_id: str, h: torch.Tensor, start_pos: int) -> torch.Tensor: ...

    @abstractmethod
    def head(self, h: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def drop_session(self, session_id: str) -> None: ...

    @abstractmethod
    def session_length(self, session_id: str) -> int: ...

    @abstractmethod
    def drop_all(self) -> None: ...

    def unload(self) -> None:  # release weights; default no-op
        return None


class ModelBackend(ABC):
    spec: ModelSpec
    tokenizer: Tokenizer

    @abstractmethod
    def load_patch(self, layers: range, *, with_embed: bool, with_head: bool, device: str = "cpu") -> PatchRuntime: ...
