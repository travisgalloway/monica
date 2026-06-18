"""MLX conversion teacher (Apple Silicon, below the seam — may import mlx).

A frozen, forward-only **Qwen2** decoder (open-r1/OpenR1-Distill-7B by default, a
Qwen2-family R1 reproduction) used as the distillation conversion teacher. It implements the portable `ConversionTeacher`
protocol (`src/model/teacher.py`) so the precompute (#94) and the distill loss (#100) see
only opaque arrays + a `to_numpy` converter, and the student init (#99) can read the
attention Q/K/V/O projections.

The weights are held as a plain `dict[str, mx.array]` rather than an `nn.Module`, so the
teacher exposes **no** parameter tree — it can never be handed to an optimizer, and
`trainable_parameters()` is empty. Forward outputs are wrapped in `mx.stop_gradient`, so
even when the teacher's logits/hidden states are composed into a student loss, no gradient
flows back into the teacher. `mlx-lm` is not a dependency; this is a minimal self-contained
forward pass, reusing the RoPE/softmax idioms from `mlx_backend`.

This file imports `mlx`, so it does NOT import on Linux/CUDA hosts — intentional and
allowed: it lives below the seam and nothing portable imports it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .teacher import (AttnProjections, ConversionTeacher, TeacherConfig,
                      TeacherForward)
from .mlx_backend import _rotate_half, _silu, _softmax_lastdim


def _rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """RMSNorm in fp32 (matches `mlx_backend.RMSNorm`)."""
    xf = x.astype(mx.float32)
    norm = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return weight * (xf * norm)


def _rope_cos_sin(positions: mx.array, head_dim: int, theta: float) -> Tuple[mx.array, mx.array]:
    """RoPE cos/sin for absolute `positions` (fp32), theta-configurable (Qwen2 rope_theta).
    Generalizes `mlx_backend._rope_cos_sin` (which hardcodes theta=10000). (T, head_dim)."""
    half = head_dim // 2
    inv_freq = mx.exp(-math.log(theta) * mx.arange(0, half, dtype=mx.float32) / half)
    ang = positions.astype(mx.float32)[:, None] * inv_freq[None, :]   # (T, half)
    cos = mx.concatenate([mx.cos(ang), mx.cos(ang)], axis=-1)         # (T, head_dim)
    sin = mx.concatenate([mx.sin(ang), mx.sin(ang)], axis=-1)
    return cos, sin


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: (B, H, T, Dh); cos/sin: (T, Dh) broadcast over (B, H).
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


class MLXConversionTeacher(ConversionTeacher):
    """Frozen Qwen2 conversion teacher on MLX. See module docstring."""

    def __init__(self, config: TeacherConfig, weights: Dict[str, mx.array]):
        config.validate()
        self.config = config
        # All compute in fp32: teacher logits feed a KL term, so stability beats speed at
        # POC scale and the teacher forward is a one-time precompute (#94) anyway.
        self._w = {k: v.astype(mx.float32) for k, v in weights.items()}

    # --- constructors --------------------------------------------------------
    @classmethod
    def from_config(cls, config: TeacherConfig, *, seed: int = 0) -> "MLXConversionTeacher":
        """Random-init synthetic teacher (offline tests / small local checks)."""
        config.validate()
        key = mx.random.key(seed)
        c = config
        scale = 1.0 / math.sqrt(c.d_model)

        def rand(shape, k):
            return mx.random.normal(shape, key=k) * scale

        keys = mx.random.split(key, 4 + c.n_layers * 9)
        ki = iter(keys)
        w: Dict[str, mx.array] = {"embed": rand((c.vocab_size, c.d_model), next(ki))}
        for i in range(c.n_layers):
            p = f"layer.{i}."
            w[p + "input_ln"] = mx.ones((c.d_model,))
            w[p + "q_w"] = rand((c.q_dim, c.d_model), next(ki))
            w[p + "q_b"] = mx.zeros((c.q_dim,))
            w[p + "k_w"] = rand((c.kv_dim, c.d_model), next(ki))
            w[p + "k_b"] = mx.zeros((c.kv_dim,))
            w[p + "v_w"] = rand((c.kv_dim, c.d_model), next(ki))
            w[p + "v_b"] = mx.zeros((c.kv_dim,))
            w[p + "o_w"] = rand((c.d_model, c.q_dim), next(ki))
            w[p + "post_ln"] = mx.ones((c.d_model,))
            w[p + "gate_w"] = rand((c.intermediate_size, c.d_model), next(ki))
            w[p + "up_w"] = rand((c.intermediate_size, c.d_model), next(ki))
            w[p + "down_w"] = rand((c.d_model, c.intermediate_size), next(ki))
        w["final_ln"] = mx.ones((c.d_model,))
        if not c.tie_embeddings:
            w["lm_head"] = rand((c.vocab_size, c.d_model), next(ki))
        mx.eval(list(w.values()))
        return cls(config, w)

    @classmethod
    def from_pretrained(cls, path, config: Optional[TeacherConfig] = None
                        ) -> "MLXConversionTeacher":
        """Load a local HuggingFace Qwen2 checkpoint directory (config.json + safetensors).

        `path` may be a local checkpoint directory or an HF repo id. When it is not a local
        path, it is treated as the repo id (so `from_pretrained("deepseek-ai/...")` works with
        no config); a bare name falls back to `config.model_id`. The fetch is lazy via
        `huggingface_hub.snapshot_download` (guarded — never reached in offline tests). HF
        row-major projection weights are mapped onto the internal dict."""
        p = Path(path)
        if not p.exists():
            # Not a local dir: use `path` as the repo id when it looks like one, else config.model_id.
            repo_id = str(path) if "/" in str(path) else (config.model_id if config else None)
            if not repo_id:
                raise FileNotFoundError(
                    f"teacher path {p} not found and no repo id (pass an HF 'owner/name' path "
                    "or a config with model_id)")
            from huggingface_hub import snapshot_download   # lazy: optional dep, not in tests
            p = Path(snapshot_download(repo_id))
        if config is None:
            with open(p / "config.json") as f:
                config = TeacherConfig.from_hf_dict(json.load(f))
        hf = _load_safetensors_dir(p)
        return cls(config, _hf_to_internal(hf, config))

    # --- ConversionTeacher ---------------------------------------------------
    def forward(self, token_batch, *, return_hidden: bool = False) -> TeacherForward:
        tokens = mx.array(token_batch)
        c = self.config
        h = self._w["embed"][tokens]                          # (B, L, D)
        L = h.shape[1]
        cos, sin = _rope_cos_sin(mx.arange(L), c.head_dim, c.rope_theta)
        mask = mx.tril(mx.ones((L, L), dtype=mx.bool_))       # causal mask, built once
        hidden: List[mx.array] = [h] if return_hidden else []
        for i in range(c.n_layers):
            h = h + self._attn(h, i, cos, sin, mask)
            h = h + self._mlp(h, i)
            if return_hidden:
                hidden.append(h)
        # Emit logits over the tokenizer vocab (padded embedding rows are never teacher targets).
        logits = (_rms_norm(h, self._w["final_ln"], c.rms_norm_eps)
                  @ self._lm_head().T)[..., :c.effective_vocab_size]
        hs = tuple(mx.stop_gradient(t) for t in hidden) if return_hidden else None
        return TeacherForward(logits=mx.stop_gradient(logits), hidden_states=hs)

    def topk_logits(self, token_batch, k: int) -> Tuple[mx.array, mx.array]:
        logits = self.forward(token_batch).logits             # (B, L, Ve)
        k = min(k, logits.shape[-1])
        # Partial selection (top-k), then sort only the k — far cheaper than a full argsort
        # over the ~152k vocab (matters for the #94 precompute).
        part = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        pv = mx.take_along_axis(logits, part, axis=-1)
        order = mx.argsort(-pv, axis=-1)                       # descending within the k
        idx = mx.take_along_axis(part, order, axis=-1)
        vals = mx.take_along_axis(pv, order, axis=-1)
        return mx.stop_gradient(vals), idx

    def attention_matrices(self, token_batch) -> Tuple[mx.array, ...]:
        """Per-layer head-averaged causal attention matrices `softmax(QK^T/sqrt(Dh))`, each
        `(B, L, L)`, for the distillation `mixing-match` stage (#100). Stop-gradient wrapped
        (the teacher is frozen)."""
        c = self.config
        h = self._w["embed"][mx.array(token_batch)]
        L = h.shape[1]
        cos, sin = _rope_cos_sin(mx.arange(L), c.head_dim, c.rope_theta)
        mask = self._causal_mask(L)                                  # built once, shared
        mats = []
        for i in range(c.n_layers):
            mats.append(mx.stop_gradient(self._attn_probs(h, i, cos, sin, mask)))
            h = h + self._attn(h, i, cos, sin, mask)
            h = h + self._mlp(h, i)
        return tuple(mats)

    def _attn_probs(self, x: mx.array, i: int, cos: mx.array, sin: mx.array,
                    mask: mx.array) -> mx.array:
        """Head-averaged causal attention probabilities for layer `i` -> (B, L, L)."""
        c = self.config
        p = f"layer.{i}."
        B, L, _ = x.shape
        Hq, Hkv, Dh = c.n_heads, c.n_kv_heads, c.head_dim
        xn = _rms_norm(x, self._w[p + "input_ln"], c.rms_norm_eps)
        q = xn @ self._w[p + "q_w"].T + self._w[p + "q_b"]
        k = xn @ self._w[p + "k_w"].T + self._w[p + "k_b"]

        def heads(t, H):
            return t.reshape(B, L, H, Dh).transpose(0, 2, 1, 3)
        q, k = heads(q, Hq), heads(k, Hkv)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        if Hkv != Hq:
            k = mx.repeat(k, Hq // Hkv, axis=1)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(Dh)       # (B,Hq,L,L)
        scores = mx.where(mask, scores, mx.array(float("-inf"), dtype=scores.dtype))
        return _softmax_lastdim(scores).mean(axis=1)                 # head-average -> (B,L,L)

    def attention_projection(self, layer: int) -> AttnProjections:
        p = f"layer.{layer}."
        g = lambda s: mx.stop_gradient(self._w[p + s])
        return AttnProjections(q=g("q_w"), k=g("k_w"), v=g("v_w"), o=g("o_w"),
                               q_bias=g("q_b"), k_bias=g("k_b"), v_bias=g("v_b"))

    def embedding_matrix(self) -> mx.array:
        return mx.stop_gradient(self._w["embed"])

    def lm_head_matrix(self) -> mx.array:
        return mx.stop_gradient(self._lm_head())   # == embed when tied (see _lm_head)

    def to_numpy(self, array) -> np.ndarray:
        return np.array(array)

    # --- internals -----------------------------------------------------------
    def _lm_head(self) -> mx.array:
        return self._w["embed"] if self.config.tie_embeddings else self._w["lm_head"]

    def _causal_mask(self, L: int) -> mx.array:
        """Causal (L, L) bool mask, built once per forward and shared across layers."""
        return mx.tril(mx.ones((L, L), dtype=mx.bool_))

    def _attn(self, x: mx.array, i: int, cos: mx.array, sin: mx.array,
              mask: mx.array) -> mx.array:
        c = self.config
        p = f"layer.{i}."
        B, L, _ = x.shape
        Hq, Hkv, Dh = c.n_heads, c.n_kv_heads, c.head_dim
        xn = _rms_norm(x, self._w[p + "input_ln"], c.rms_norm_eps)
        q = xn @ self._w[p + "q_w"].T + self._w[p + "q_b"]    # (B,L,q_dim)
        k = xn @ self._w[p + "k_w"].T + self._w[p + "k_b"]    # (B,L,kv_dim)
        v = xn @ self._w[p + "v_w"].T + self._w[p + "v_b"]

        def heads(t, H):
            return t.reshape(B, L, H, Dh).transpose(0, 2, 1, 3)   # (B,H,L,Dh)
        q, k, v = heads(q, Hq), heads(k, Hkv), heads(v, Hkv)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        if Hkv != Hq:                                         # GQA: repeat kv across groups
            rep = Hq // Hkv
            k, v = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(Dh)   # (B,Hq,L,L)
        scores = mx.where(mask, scores, mx.array(float("-inf"), dtype=scores.dtype))
        out = _softmax_lastdim(scores) @ v                   # (B,Hq,L,Dh)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, Hq * Dh)
        return out @ self._w[p + "o_w"].T                    # (B,L,D), o has no bias

    def _mlp(self, x: mx.array, i: int) -> mx.array:
        c = self.config
        p = f"layer.{i}."
        xn = _rms_norm(x, self._w[p + "post_ln"], c.rms_norm_eps)
        gate = _silu(xn @ self._w[p + "gate_w"].T)
        up = xn @ self._w[p + "up_w"].T
        return (gate * up) @ self._w[p + "down_w"].T


# --- HuggingFace checkpoint loading ------------------------------------------
def _load_safetensors_dir(p: Path) -> Dict[str, mx.array]:
    """Load + merge every `*.safetensors` shard in a directory into one dict."""
    files = sorted(p.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors found in {p}")
    out: Dict[str, mx.array] = {}
    for f in files:
        out.update(mx.load(str(f)))
    return out


def _hf_to_internal(hf: Dict[str, mx.array], cfg: TeacherConfig) -> Dict[str, mx.array]:
    """Map HuggingFace Qwen2 parameter names onto the internal weight-dict layout."""
    w: Dict[str, mx.array] = {"embed": hf["model.embed_tokens.weight"]}
    for i in range(cfg.n_layers):
        hp, p = f"model.layers.{i}.", f"layer.{i}."
        w[p + "input_ln"] = hf[hp + "input_layernorm.weight"]
        w[p + "post_ln"] = hf[hp + "post_attention_layernorm.weight"]
        for proj, dst in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
            w[p + dst + "_w"] = hf[hp + f"self_attn.{proj}.weight"]
            w[p + dst + "_b"] = hf[hp + f"self_attn.{proj}.bias"]
        w[p + "o_w"] = hf[hp + "self_attn.o_proj.weight"]
        w[p + "gate_w"] = hf[hp + "mlp.gate_proj.weight"]
        w[p + "up_w"] = hf[hp + "mlp.up_proj.weight"]
        w[p + "down_w"] = hf[hp + "mlp.down_proj.weight"]
    w["final_ln"] = hf["model.norm.weight"]
    if not cfg.tie_embeddings:
        w["lm_head"] = hf["lm_head.weight"]
    return w
