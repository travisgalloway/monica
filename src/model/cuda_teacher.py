"""CUDA / PyTorch conversion teacher (below the seam — may import torch).

A faithful torch port of `mlx_teacher.MLXConversionTeacher`: a frozen, forward-only Qwen2
decoder (open-r1/OpenR1-Distill-7B by default) implementing the portable `ConversionTeacher`
protocol (`src/model/teacher.py`). It exists so the dominant teacher precompute (#94) can run
on the cloud GPU via `--backend cuda`; the MLX teacher remains the Apple-Silicon dev path.

Like the MLX teacher, weights are a plain `dict[str, torch.Tensor]` (NOT an `nn.Module`), so
the teacher exposes no parameter tree — it can never be handed to an optimizer, and
`trainable_parameters()` is empty. Forward outputs are `.detach()`-ed (the torch analogue of
`mx.stop_gradient`): even composed into a student loss, no gradient reaches the teacher. All
compute is fp32 (teacher logits feed a KL term; stability beats speed and the forward is a
one-time precompute). Runs on CUDA when available, else CPU (so parity is testable on a Mac/Linux
box). RoPE/attention idioms mirror `cuda_backend.py` but with a theta-configurable RoPE (Qwen2
`rope_theta`, up to 300k for Open-R1's 32k context).

This file imports `torch`, so it lives BELOW the seam — nothing portable imports it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .teacher import (AttnProjections, ConversionTeacher, TeacherConfig,
                      TeacherForward)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32 (matches `cuda_backend.RMSNorm` / `mlx_teacher._rms_norm`)."""
    xf = x.to(torch.float32)
    norm = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    return weight * (xf * norm)


def _rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE cos/sin for absolute `positions` (fp32), theta-configurable (Qwen2 rope_theta).
    Generalizes `cuda_backend._rope_cos_sin` (which hardcodes theta=10000). (T, head_dim)."""
    half = head_dim // 2
    device = positions.device
    inv_freq = torch.exp(-math.log(theta)
                         * torch.arange(0, half, dtype=torch.float32, device=device) / half)
    ang = positions.to(torch.float32)[:, None] * inv_freq[None, :]    # (T, half)
    cos = torch.cat([torch.cos(ang), torch.cos(ang)], dim=-1)         # (T, head_dim)
    sin = torch.cat([torch.sin(ang), torch.sin(ang)], dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, Dh); cos/sin: (T, Dh) broadcast over (B, H).
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


def _softmax_lastdim(scores: torch.Tensor) -> torch.Tensor:
    scores = scores - torch.amax(scores, dim=-1, keepdim=True)
    w = torch.exp(scores)
    return w / torch.sum(w, dim=-1, keepdim=True)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


class CUDATeacher(ConversionTeacher):
    """Frozen Qwen2 conversion teacher on PyTorch. See module docstring."""

    def __init__(self, config: TeacherConfig, weights: Dict[str, torch.Tensor]):
        config.validate()
        self.config = config
        self._device = _device()
        # All compute in fp32 (KL stability; one-time precompute). Held off the autograd graph.
        self._w = {k: v.detach().to(self._device, torch.float32) for k, v in weights.items()}

    # --- constructors --------------------------------------------------------
    @classmethod
    def from_config(cls, config: TeacherConfig, *, seed: int = 0) -> "CUDATeacher":
        """Random-init synthetic teacher (offline tests / small local checks). The weight
        layout matches `mlx_teacher.from_config` so a parity test can build both from the
        same numpy arrays."""
        config.validate()
        g = torch.Generator().manual_seed(seed)
        c = config
        scale = 1.0 / math.sqrt(c.d_model)

        def rand(*shape):
            return torch.randn(*shape, generator=g) * scale

        w: Dict[str, torch.Tensor] = {"embed": rand(c.vocab_size, c.d_model)}
        for i in range(c.n_layers):
            p = f"layer.{i}."
            w[p + "input_ln"] = torch.ones(c.d_model)
            w[p + "q_w"] = rand(c.q_dim, c.d_model)
            w[p + "q_b"] = torch.zeros(c.q_dim)
            w[p + "k_w"] = rand(c.kv_dim, c.d_model)
            w[p + "k_b"] = torch.zeros(c.kv_dim)
            w[p + "v_w"] = rand(c.kv_dim, c.d_model)
            w[p + "v_b"] = torch.zeros(c.kv_dim)
            w[p + "o_w"] = rand(c.d_model, c.q_dim)
            w[p + "post_ln"] = torch.ones(c.d_model)
            w[p + "gate_w"] = rand(c.intermediate_size, c.d_model)
            w[p + "up_w"] = rand(c.intermediate_size, c.d_model)
            w[p + "down_w"] = rand(c.d_model, c.intermediate_size)
        w["final_ln"] = torch.ones(c.d_model)
        if not c.tie_embeddings:
            w["lm_head"] = rand(c.vocab_size, c.d_model)
        return cls(config, w)

    @classmethod
    def from_pretrained(cls, path, config: Optional[TeacherConfig] = None) -> "CUDATeacher":
        """Load a local HuggingFace Qwen2 checkpoint (config.json + safetensors), or an HF repo id.

        Mirrors `mlx_teacher.from_pretrained`: a bare 'owner/name' is fetched lazily via
        `huggingface_hub.snapshot_download`; `config.json` resolves to a `TeacherConfig` when
        none is passed (but pass `openr1_distill_7b()` so logits are sliced to the tokenizer
        vocab — see that config's note)."""
        p = Path(path)
        if not p.exists():
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
        tokens = torch.as_tensor(np.asarray(token_batch), dtype=torch.long, device=self._device)
        c = self.config
        h = self._w["embed"][tokens]                          # (B, L, D)
        L = h.shape[1]
        cos, sin = _rope_cos_sin(torch.arange(L, device=self._device), c.head_dim, c.rope_theta)
        mask = self._causal_mask(L)
        hidden: List[torch.Tensor] = [h] if return_hidden else []
        for i in range(c.n_layers):
            h = h + self._attn(h, i, cos, sin, mask)
            h = h + self._mlp(h, i)
            if return_hidden:
                hidden.append(h)
        # Emit logits over the tokenizer vocab (padded embedding rows are never teacher targets).
        logits = (_rms_norm(h, self._w["final_ln"], c.rms_norm_eps)
                  @ self._lm_head().t())[..., :c.effective_vocab_size]
        hs = tuple(t.detach() for t in hidden) if return_hidden else None
        return TeacherForward(logits=logits.detach(), hidden_states=hs)

    def topk_logits(self, token_batch, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(token_batch).logits             # (B, L, Ve)
        k = min(k, logits.shape[-1])
        vals, idx = torch.topk(logits, k, dim=-1)             # sorted descending
        return vals.detach(), idx

    def attention_matrices(self, token_batch) -> Tuple[torch.Tensor, ...]:
        """Per-layer head-averaged causal attention `softmax(QK^T/sqrt(Dh))`, each `(B, L, L)`,
        for the distillation `mixing-match` stage (#100). Detached (frozen teacher)."""
        c = self.config
        tokens = torch.as_tensor(np.asarray(token_batch), dtype=torch.long, device=self._device)
        h = self._w["embed"][tokens]
        L = h.shape[1]
        cos, sin = _rope_cos_sin(torch.arange(L, device=self._device), c.head_dim, c.rope_theta)
        mask = self._causal_mask(L)
        mats = []
        for i in range(c.n_layers):
            mats.append(self._attn_probs(h, i, cos, sin, mask).detach())
            h = h + self._attn(h, i, cos, sin, mask)
            h = h + self._mlp(h, i)
        return tuple(mats)

    def attention_projection(self, layer: int) -> AttnProjections:
        p = f"layer.{layer}."
        g = lambda s: self._w[p + s].detach()
        return AttnProjections(q=g("q_w"), k=g("k_w"), v=g("v_w"), o=g("o_w"),
                               q_bias=g("q_b"), k_bias=g("k_b"), v_bias=g("v_b"))

    def embedding_matrix(self) -> torch.Tensor:
        return self._w["embed"].detach()

    def lm_head_matrix(self) -> torch.Tensor:
        return self._lm_head().detach()

    def to_numpy(self, array) -> np.ndarray:
        return array.detach().to("cpu").numpy()

    # --- internals -----------------------------------------------------------
    def _lm_head(self) -> torch.Tensor:
        return self._w["embed"] if self.config.tie_embeddings else self._w["lm_head"]

    def _causal_mask(self, L: int) -> torch.Tensor:
        return torch.tril(torch.ones((L, L), dtype=torch.bool, device=self._device))

    def _attn_probs(self, x: torch.Tensor, i: int, cos: torch.Tensor, sin: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """Head-averaged causal attention probabilities for layer `i` -> (B, L, L)."""
        c = self.config
        p = f"layer.{i}."
        B, L, _ = x.shape
        Hq, Hkv, Dh = c.n_heads, c.n_kv_heads, c.head_dim
        xn = _rms_norm(x, self._w[p + "input_ln"], c.rms_norm_eps)
        q = xn @ self._w[p + "q_w"].t() + self._w[p + "q_b"]
        k = xn @ self._w[p + "k_w"].t() + self._w[p + "k_b"]

        def heads(t, H):
            return t.reshape(B, L, H, Dh).permute(0, 2, 1, 3)
        q, k = heads(q, Hq), heads(k, Hkv)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        if Hkv != Hq:
            k = k.repeat_interleave(Hq // Hkv, dim=1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(Dh)       # (B,Hq,L,L)
        scores = scores.masked_fill(~mask, float("-inf"))
        return _softmax_lastdim(scores).mean(dim=1)              # head-average -> (B,L,L)

    def _attn(self, x: torch.Tensor, i: int, cos: torch.Tensor, sin: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
        c = self.config
        p = f"layer.{i}."
        B, L, _ = x.shape
        Hq, Hkv, Dh = c.n_heads, c.n_kv_heads, c.head_dim
        xn = _rms_norm(x, self._w[p + "input_ln"], c.rms_norm_eps)
        q = xn @ self._w[p + "q_w"].t() + self._w[p + "q_b"]    # (B,L,q_dim)
        k = xn @ self._w[p + "k_w"].t() + self._w[p + "k_b"]    # (B,L,kv_dim)
        v = xn @ self._w[p + "v_w"].t() + self._w[p + "v_b"]

        def heads(t, H):
            return t.reshape(B, L, H, Dh).permute(0, 2, 1, 3)   # (B,H,L,Dh)
        q, k, v = heads(q, Hq), heads(k, Hkv), heads(v, Hkv)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        if Hkv != Hq:                                          # GQA: repeat kv across groups
            rep = Hq // Hkv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(Dh)      # (B,Hq,L,L)
        scores = scores.masked_fill(~mask, float("-inf"))
        out = _softmax_lastdim(scores) @ v                     # (B,Hq,L,Dh)
        out = out.permute(0, 2, 1, 3).reshape(B, L, Hq * Dh)
        return out @ self._w[p + "o_w"].t()                    # (B,L,D), o has no bias

    def _mlp(self, x: torch.Tensor, i: int) -> torch.Tensor:
        c = self.config
        p = f"layer.{i}."
        xn = _rms_norm(x, self._w[p + "post_ln"], c.rms_norm_eps)
        gate = _silu(xn @ self._w[p + "gate_w"].t())
        up = xn @ self._w[p + "up_w"].t()
        return (gate * up) @ self._w[p + "down_w"].t()


# --- HuggingFace checkpoint loading ------------------------------------------
def _load_safetensors_dir(p: Path) -> Dict[str, torch.Tensor]:
    """Load + merge every `*.safetensors` shard in a directory into one dict."""
    from safetensors.torch import load_file
    files = sorted(p.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors found in {p}")
    out: Dict[str, torch.Tensor] = {}
    for f in files:
        out.update(load_file(str(f)))
    return out


def _hf_to_internal(hf: Dict[str, torch.Tensor], cfg: TeacherConfig) -> Dict[str, torch.Tensor]:
    """Map HuggingFace Qwen2 parameter names onto the internal weight-dict layout (mirrors
    `mlx_teacher._hf_to_internal`)."""
    w: Dict[str, torch.Tensor] = {"embed": hf["model.embed_tokens.weight"]}
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
