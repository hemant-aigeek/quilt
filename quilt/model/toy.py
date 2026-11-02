"""A small, self-contained decoder-only transformer with deterministic weights.

It exists so the whole system — planning, pipelining, failover — can be run
and tested on any machine with no model download. Weights come from a seeded
generator, so every member derives *identical* parameters from the config
alone; a byte-level tokenizer means any text is representable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..planner import ModelSpec
from .base import ModelBackend, PatchRuntime


@dataclass(frozen=True)
class ToyConfig:
    hidden: int = 64
    heads: int = 4
    layers: int = 8
    vocab: int = 256
    max_seq: int = 1024
    seed: int = 7
    name: str = "toy"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class ToyBlock(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.heads = cfg.heads
        self.hd = cfg.hidden // cfg.heads
        self.n1 = RMSNorm(cfg.hidden)
        self.qkv = nn.Linear(cfg.hidden, 3 * cfg.hidden, bias=False)
        self.o = nn.Linear(cfg.hidden, cfg.hidden, bias=False)
        self.n2 = RMSNorm(cfg.hidden)
        self.up = nn.Linear(cfg.hidden, 4 * cfg.hidden, bias=False)
        self.down = nn.Linear(4 * cfg.hidden, cfg.hidden, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int, cache: tuple[torch.Tensor, torch.Tensor] | None):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.hd).transpose(1, 2)
        k = k.view(b, t, self.heads, self.hd).transpose(1, 2)
        v = v.view(b, t, self.heads, self.hd).transpose(1, 2)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        total = k.shape[2]
        # causal mask over absolute positions: query i (at start_pos+i) sees keys <= start_pos+i
        qpos = torch.arange(start_pos, start_pos + t, device=x.device)[:, None]
        kpos = torch.arange(total, device=x.device)[None, :]
        mask = kpos <= qpos
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, -1))
        x = x + self.down(F.silu(self.up(self.n2(x))))
        return x, (k, v)


class ByteTokenizer:
    eos_id = None

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i & 0xFF for i in ids).decode("utf-8", errors="replace")


def _seeded_init(module: nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for _name, p in sorted(module.named_parameters()):
        if p.dim() >= 2:
            fan_in = p.shape[1]
            std = 1.0 / math.sqrt(fan_in)
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * std)


class ToyPatch(PatchRuntime):
    def __init__(self, cfg: ToyConfig, layers: range, with_embed: bool, with_head: bool, device: str):
        self.cfg = cfg
        self.layers = layers
        self.has_embed = with_embed
        self.has_head = with_head
        self.device = device
        self.blocks = nn.ModuleList()
        for i in layers:
            blk = ToyBlock(cfg)
            _seeded_init(blk, cfg.seed * 1000 + i)
            self.blocks.append(blk)
        self.tok_emb = self.pos_emb = self.norm = self.lm_head = None
        if with_embed:
            self.tok_emb = nn.Embedding(cfg.vocab, cfg.hidden)
            self.pos_emb = nn.Embedding(cfg.max_seq, cfg.hidden)
            _seeded_init(self.tok_emb, cfg.seed * 1000 + 900)
            _seeded_init(self.pos_emb, cfg.seed * 1000 + 901)
        if with_head:
            self.norm = RMSNorm(cfg.hidden)
            self.lm_head = nn.Linear(cfg.hidden, cfg.vocab, bias=False)
            _seeded_init(self.lm_head, cfg.seed * 1000 + 902)
        self.blocks.to(device).eval()
        for m in (self.tok_emb, self.pos_emb, self.norm, self.lm_head):
            if m is not None:
                m.to(device).eval()
        self._cache: dict[str, list[tuple[torch.Tensor, torch.Tensor] | None]] = {}

    @torch.inference_mode()
    def embed(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        assert self.has_embed
        tokens = tokens.to(self.device)
        pos = torch.arange(start_pos, start_pos + tokens.shape[1], device=self.device)
        return self.tok_emb(tokens) + self.pos_emb(pos)[None]

    @torch.inference_mode()
    def forward(self, session_id: str, h: torch.Tensor, start_pos: int) -> torch.Tensor:
        h = h.to(self.device)
        caches = self._cache.setdefault(session_id, [None] * len(self.blocks))
        if start_pos != self.session_length(session_id):
            # Positions disagree with what we cached: rebuild from scratch.
            caches = self._cache[session_id] = [None] * len(self.blocks)
            if start_pos != 0:
                raise ValueError(f"session {session_id}: cache has {self.session_length(session_id)} "
                                 f"positions but step starts at {start_pos}")
        for i, blk in enumerate(self.blocks):
            h, caches[i] = blk(h, start_pos, caches[i])
        return h

    @torch.inference_mode()
    def head(self, h: torch.Tensor) -> torch.Tensor:
        assert self.has_head
        return self.lm_head(self.norm(h.to(self.device)))

    def drop_session(self, session_id: str) -> None:
        self._cache.pop(session_id, None)

    def drop_all(self) -> None:
        self._cache.clear()

    def session_length(self, session_id: str) -> int:
        c = self._cache.get(session_id)
        if not c or c[0] is None:
            return 0
        return int(c[0][0].shape[2])


class ToyBackend(ModelBackend):
    def __init__(self, cfg: ToyConfig | None = None):
        self.cfg = cfg or ToyConfig()
        c = self.cfg
        per_layer = (3 * c.hidden * c.hidden + c.hidden * c.hidden + 8 * c.hidden * c.hidden + 2 * c.hidden) * 4
        kv = 2 * c.max_seq * c.hidden * 4
        self.spec = ModelSpec(
            name=c.name, n_layers=c.layers, hidden=c.hidden, heads=c.heads, vocab=c.vocab, max_seq=c.max_seq,
            bytes_per_layer=per_layer, embed_bytes=(c.vocab + c.max_seq) * c.hidden * 4,
            head_bytes=c.vocab * c.hidden * 4 + c.hidden * 4, kv_bytes_per_layer=kv,
        )
        self.tokenizer = ByteTokenizer()

    def load_patch(self, layers: range, *, with_embed: bool, with_head: bool, device: str = "cpu") -> ToyPatch:
        return ToyPatch(self.cfg, layers, with_embed, with_head, device)
