"""Hugging Face backend for Llama-family checkpoints (Llama, Qwen2, Mistral…).

Only the layers a member owns are kept resident: the checkpoint is loaded
with ``low_cpu_mem_usage`` and everything outside the patch is dropped before
the first forward pass. Each session gets its own ``DynamicCache``.

Requires ``transformers >= 4.45``. The decoder-layer calling convention has
shifted across releases; ``_call_layer`` normalises the two common shapes.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch

from ..planner import ModelSpec
from .base import ModelBackend, PatchRuntime


@dataclass
class HFConfig:
    model_id: str
    dtype: str = "float16"
    trust_remote_code: bool = False
    max_seq: int = 4096


class HFTokenizer:
    def __init__(self, tok):
        self._t = tok

    def encode(self, text: str) -> list[int]:
        return self._t.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._t.decode(ids, skip_special_tokens=True)

    @property
    def eos_id(self) -> int | None:
        return self._t.eos_token_id


class HFPatch(PatchRuntime):
    def __init__(self, cfg: HFConfig, layers: range, with_embed: bool, with_head: bool, device: str):
        from transformers import AutoModelForCausalLM, DynamicCache

        self._DynamicCache = DynamicCache
        self.layers = layers
        self.has_embed = with_embed
        self.has_head = with_head
        self.device = device
        dtype = getattr(torch, cfg.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=cfg.trust_remote_code
        )
        inner = model.model
        self.rotary = getattr(inner, "rotary_emb", None)
        self.blocks = torch.nn.ModuleList([inner.layers[i] for i in layers])
        self.embed_tokens = inner.embed_tokens if with_embed else None
        self.norm = inner.norm if with_head else None
        self.lm_head = model.lm_head if with_head else None
        # Drop everything else so memory reflects the patch, not the checkpoint.
        inner.layers = torch.nn.ModuleList()
        del model, inner
        gc.collect()
        for m in (self.blocks, self.embed_tokens, self.norm, self.lm_head, self.rotary):
            if m is not None:
                m.to(device).eval()
        self._cache: dict[str, object] = {}
        self._len: dict[str, int] = {}

    @torch.inference_mode()
    def embed(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        return self.embed_tokens(tokens.to(self.device))

    def _call_layer(self, blk, h, position_ids, cache, cache_position, position_embeddings):
        import inspect

        params = inspect.signature(blk.forward).parameters
        kwargs = {
            "attention_mask": None,
            "position_ids": position_ids,
            "use_cache": True,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            # the cache kwarg was renamed across transformers releases
            "past_key_values" if "past_key_values" in params else "past_key_value": cache,
        }
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in params}
        out = blk(h, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    @torch.inference_mode()
    def forward(self, session_id: str, h: torch.Tensor, start_pos: int) -> torch.Tensor:
        h = h.to(self.device)
        if start_pos != self._len.get(session_id, 0):
            if start_pos != 0:
                raise ValueError(f"session {session_id}: cache at {self._len.get(session_id, 0)}, step at {start_pos}")
            self._cache.pop(session_id, None)
            self._len[session_id] = 0
        cache = self._cache.setdefault(session_id, self._DynamicCache())
        t = h.shape[1]
        position_ids = torch.arange(start_pos, start_pos + t, device=self.device)[None]
        position_embeddings = self.rotary(h, position_ids) if self.rotary is not None else None
        for blk in self.blocks:
            h = self._call_layer(blk, h, position_ids, cache, position_ids[0], position_embeddings)
        self._len[session_id] = start_pos + t
        return h

    @torch.inference_mode()
    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(h.to(self.device)))

    def drop_session(self, session_id: str) -> None:
        self._cache.pop(session_id, None)
        self._len.pop(session_id, None)

    def session_length(self, session_id: str) -> int:
        return self._len.get(session_id, 0)

    def drop_all(self) -> None:
        self._cache.clear()
        self._len.clear()

    def unload(self) -> None:
        self.blocks = torch.nn.ModuleList()
        self.embed_tokens = self.norm = self.lm_head = None
        gc.collect()


class HFBackend(ModelBackend):
    def __init__(self, cfg: HFConfig):
        from transformers import AutoConfig, AutoTokenizer

        self.cfg = cfg
        c = AutoConfig.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
        hidden, inter = c.hidden_size, getattr(c, "intermediate_size", 4 * c.hidden_size)
        heads = c.num_attention_heads
        kv_heads = getattr(c, "num_key_value_heads", heads)
        head_dim = hidden // heads
        bpe = torch.tensor([], dtype=getattr(torch, cfg.dtype)).element_size()
        attn = hidden * (hidden + 2 * kv_heads * head_dim) + hidden * hidden
        mlp = 3 * hidden * inter
        self.spec = ModelSpec(
            name=cfg.model_id, n_layers=c.num_hidden_layers, hidden=hidden, heads=heads, vocab=c.vocab_size,
            max_seq=min(cfg.max_seq, getattr(c, "max_position_embeddings", cfg.max_seq)),
            bytes_per_layer=(attn + mlp + 2 * hidden) * bpe,
            embed_bytes=c.vocab_size * hidden * bpe,
            head_bytes=c.vocab_size * hidden * bpe,
            kv_bytes_per_layer=2 * kv_heads * head_dim * min(cfg.max_seq, 4096) * bpe,
        )
        self.tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code))

    def load_patch(self, layers: range, *, with_embed: bool, with_head: bool, device: str = "cpu") -> HFPatch:
        return HFPatch(self.cfg, layers, with_embed, with_head, device)
