"""Sparse-upcycle: turn a dense (degenerate `n_experts=1`) MoE checkpoint into a real
`n_experts>1` MoE checkpoint that computes the SAME function at step 0 (#214).

Portable — numpy + stdlib + `..model.blocks` only, NEVER `mlx` or `torch` (see
`tests/test_import_guard.py::PORTABLE_MODULES`). This is what lets `scripts/upcycle.py`
stay a standalone, offline, no-backend tool: the transform operates on plain
`{name: np.ndarray}` dicts, exactly the shape `checkpoint.load_weights_dict` returns.

Why the source must be `n_experts=1, top_k=1`, not a plain non-MoE config: at E=1,
`softmax` over one logit is exactly 1.0 and `MoEBlock._moe` takes the `k == E` branch —
the degenerate MoE block IS a plain SwiGLU FFN (see `MambaConfig.validate()`'s
`n_experts == 1` carve-out). So #200's dense checkpoint is trained as a "1-expert MoE"
from the start, and this module's whole trick is: replicate that one expert into every
slot of the real (E>1) target, so every token's top_k selection sees `n_experts` COPIES
of the function it was already computing. Combined with a fresh, additively-neutral
router seed (below), the target reproduces the source's logits at step 0 EXACTLY, up to
float rounding.

**Deliberately NOT implemented: upcycling from a source with `n_experts > 1`.** Step-0
exactness does not survive it — the target's top_k selection over N replicas of each of
E>1 distinct experts does not select the same subset of "logical" experts the source's
top_k selected, so the combined output changes. The E=1 restriction is not an
implementation gap; it is the only case where this transform is exact.

Also NOT implemented: width-changing upcycle (source `d_model` != target `d_model`).
Expert replication only ADDS expert slots at a FIXED width — it cannot widen the
existing tensors. Widening is Net2Net / bert2BERT-style function-preserving width
expansion, a materially different technique this module does not implement. See the
`d_model` mismatch branch of `check_upcycle_compatible` and the open #223 issue (Large
A's `d_model` relative to the dense checkpoint's is not yet decided).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from ..model.blocks import MambaConfig

# Fields that MUST agree between the source and target config for the transform to be
# well-defined. Deliberately compares RESOLVED properties (`dt_rank_resolved`, `d_inner`,
# `n_heads`, `n_attn_heads_resolved`, `moe_d_ff_resolved`) rather than the raw
# `Optional[str|int]` fields (`dt_rank`, `moe_d_ff`) they derive from, so `"auto"`/`None`
# on one side and an equal concrete value on the other compare equal — exactly how the
# two configs' TENSORS would compare.
_MUST_MATCH = (
    "d_model", "n_layers", "d_state", "expand", "d_conv", "head_dim",
    "dt_rank_resolved", "d_inner", "n_heads", "vocab_size", "tie_embeddings",
    "attn_every", "n_attn_heads_resolved", "moe_every", "moe_d_ff_resolved",
)

_EXPERT_LINS = ("gate", "up", "down")
# Kept in lockstep with `MLXMambaModel._portable_state_dict` / `CUDAMambaModel.
# _portable_state_dict` (verified byte-identical between the two backends). Not a
# parameter — see `MoEBlock.__init__` in either backend for why the leading underscore
# excludes it from `parameters()`/`named_parameters()`.
_MOE_BIAS_PREFIX = "moe_route_bias."


class UpcycleError(ValueError):
    """Raised for any source/target config mismatch or malformed source checkpoint that
    would make the upcycle transform ill-defined."""


def check_upcycle_compatible(src_cfg: MambaConfig, dst_cfg: MambaConfig) -> None:
    """Raise `UpcycleError` unless `src_cfg` is a valid dense-upcycle source for
    `dst_cfg`. Reports EVERY mismatch at once (not just the first) — one-at-a-time
    raises are exactly how a run ends up mis-dimensioned (#223): a fix-one/re-run loop
    can silently stop after clearing the first error while a second, unrelated mismatch
    is still live.
    """
    if not (src_cfg.n_experts == 1 and src_cfg.top_k == 1):
        raise UpcycleError(
            f"upcycle source must be a degenerate n_experts=1/top_k=1 MoE block (a "
            f"plain dense SwiGLU FFN in disguise — see the module docstring), got "
            f"n_experts={src_cfg.n_experts} top_k={src_cfg.top_k}. An ALREADY-MoE "
            "checkpoint is not a valid upcycle source "
            "(docs/design/13-code-model-moe.md:234-255): start from #200's dense "
            "checkpoint (trained with n_experts=1, top_k=1) instead."
        )

    mismatches = []
    for name in _MUST_MATCH:
        sv, dv = getattr(src_cfg, name), getattr(dst_cfg, name)
        if sv != dv:
            mismatches.append((name, sv, dv))

    src_moe = {i for i in range(src_cfg.n_layers) if src_cfg.is_moe_layer(i)}
    dst_moe = {i for i in range(dst_cfg.n_layers) if dst_cfg.is_moe_layer(i)}
    moe_set_mismatch = src_moe != dst_moe
    if moe_set_mismatch:
        mismatches.append(("moe_layer_indices", sorted(src_moe), sorted(dst_moe)))

    if not mismatches:
        return

    lines = [f"  {name}: src={sv!r} dst={dv!r}" for name, sv, dv in mismatches]
    msg = ("upcycle requires the source and target configs to agree on every "
           "non-expert dimension; found mismatch(es):\n" + "\n".join(lines))
    if any(name == "d_model" for name, _, _ in mismatches):
        # A dedicated paragraph, not just another bullet: this is the one mismatch that
        # is NOT a config-fixing exercise — expert replication (this module) cannot
        # widen tensors at all.
        msg += (
            "\n\nd_model differs between source and target: expert REPLICATION (what "
            "this module does) can only add expert SLOTS at a fixed width — it cannot "
            "widen the underlying tensors. Widening a checkpoint is Net2Net / "
            "bert2BERT-style function-preserving width expansion, a different "
            "technique that is not implemented here. This is the open #223 question "
            "(Large A's d_model relative to the dense source) — resolve it there "
            "before attempting a width-changing upcycle."
        )
    raise UpcycleError(msg)


def upcycle_dense_to_moe(weights: Dict[str, np.ndarray], src_cfg: MambaConfig,
                         dst_cfg: MambaConfig, *, seed: int,
                         router_init_scale: float = 1.0,
                         shared_expert_init: str = "zero_down") -> Dict[str, np.ndarray]:
    """Transform a dense (`src_cfg`, `n_experts=1`) portable weight dict into a real MoE
    (`dst_cfg`, `n_experts>1`) one, exact at step 0.

    Per MoE layer (layer indices come from `dst_cfg.is_moe_layer`, NEVER from key-pattern
    sniffing — the configs are the source of truth; key patterns are used only to
    validate the source actually has what `src_cfg` claims):

      * Experts — `.copy()` `experts.0.{gate,up,down}.weight` into all `dst_cfg.n_experts`
        slots. `.copy()`, not a shared reference: `load_weights_dict` returns
        safetensors-backed arrays that may be read-only mmaps, and even when they are
        not, aliasing would make an in-place training update to one "expert" silently
        mutate all of them.
      * Router — the source's `[1, d_model]` router is DISCARDED (a 1-row router carries
        no routing information to preserve) and replaced with a fresh
        `U(-router_init_scale/sqrt(d_model), +router_init_scale/sqrt(d_model))` draw from
        `np.random.default_rng(seed)`, filled in ascending MoE-layer-index order for
        reproducibility. That bound is exactly what both backends' `nn.Linear` init
        already produces at `router_init_scale=1.0` (MLX: `U(±1/sqrt(fan_in))`; torch:
        `kaiming_uniform_(a=sqrt(5))` reduces to the same bound for a linear layer) — so
        this introduces no new magic constant. Router init does not threaten step-0
        exactness: EVERY expert computes the identical function (copied from the same
        source expert), so no matter which top_k the fresh router selects, or what
        renormalized gate weights it assigns them, the weighted combination of top_k
        IDENTICAL functions equals that same function again (weights sum to 1 after
        renormalization).
      * Shared experts (`dst_cfg.n_shared_experts`) — DeepSeek-V2/V3-style, additive and
        OUTSIDE the router softmax (see `MoEBlock._moe`). For an index `j` present in
        BOTH configs (`j < src_cfg.n_shared_experts`), the source's shared expert is
        copied through unchanged. For a NEW index (`j >= src_cfg.n_shared_experts`,
        including the common `n_shared_experts: 0 -> 1` case), the default
        `shared_expert_init="zero_down"` draws a fresh `gate`/`up` at the same
        `U(±1/sqrt(fan_in))` bound and sets `down` to EXACT ZEROS. A shared expert
        contributes `down(silu(gate(x)) * up(x))`; zeroing `down` makes the WHOLE
        expression identically zero regardless of `gate`/`up`, which is what makes step-0
        exactness hold REGARDLESS of how the block combines the shared and routed paths
        (additive, gated, pre- or post-norm — whatever the combination formula turns out
        to be). This is why zeroing `down` beats the naive "split the dense FFN width in
        half between routed and shared": that alternative is also exact, but only under
        the SPECIFIC combination formula it was derived for, and silently wrong under any
        other. `shared_expert_init="forbid"` raises instead of synthesizing a new shared
        expert (the escape hatch for a caller that wants an error, not a guess, when the
        configs disagree on shared-expert count).
      * `moe_route_bias.*` — dropped unconditionally, from every layer. PROVABLY lossless
        at `n_experts=1`: `MoEBalancer.update` moves each layer's bias by
        `rate * sign(mean(load) - load_i)`; with exactly one expert, `load_i IS the
        mean(load)` for every step, so `sign(0) == 0` and the bias is 0 forever (also
        enforced structurally: `MambaConfig.validate()` REQUIRES `moe_balance_rate is
        None` whenever `n_experts == 1`, so a #214-era dense source could not have a
        nonzero bias to begin with). Even if a stray length-1 bias vector were present,
        it could not be broadcast onto a length-`dst_cfg.n_experts` router bias at any
        `n_experts > 1` — so dropping it is both correct and the only option.

    Every OTHER key (embedding, non-MoE layers, norms, the LM head if untied) is copied
    through unchanged; `check_upcycle_compatible` having already required identical
    non-expert dimensions and an identical MoE-layer-index SET means a non-MoE-layer
    key's name and shape are identical in both configs by construction.

    Raises `UpcycleError` (via `check_upcycle_compatible`) on any config mismatch, on a
    source checkpoint missing an expected `experts.0.*` key, on an unknown
    `shared_expert_init`, or if the final output's key set/shapes don't exactly match
    `dst_cfg`'s expected layout (an internal consistency assertion, not something a
    caller should be able to trigger with a valid source+configs).
    """
    check_upcycle_compatible(src_cfg, dst_cfg)
    if shared_expert_init not in ("zero_down", "forbid"):
        raise UpcycleError(
            f"unknown shared_expert_init {shared_expert_init!r} "
            "(expected 'zero_down' or 'forbid')"
        )

    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {
        k: v for k, v in weights.items() if not k.startswith(_MOE_BIAS_PREFIX)
    }

    d_model = dst_cfg.d_model
    d_ff = dst_cfg.moe_d_ff_resolved
    router_bound = router_init_scale / math.sqrt(d_model)
    gate_up_bound = 1.0 / math.sqrt(d_model)          # fan_in of gate/up is d_model

    moe_idx = sorted(i for i in range(dst_cfg.n_layers) if dst_cfg.is_moe_layer(i))
    for i in moe_idx:
        prefix = f"layers.{i}."
        src_expert = {}
        for lin in _EXPERT_LINS:
            key = f"{prefix}experts.0.{lin}.weight"
            if key not in weights:
                raise UpcycleError(
                    f"source weights are missing {key!r} — layer {i} is a MoE layer in "
                    "both configs, so the source (n_experts=1) must carry a single "
                    "experts.0.* expert there."
                )
            src_expert[lin] = np.asarray(weights[key])

        # Drop whatever expert keys the source had at this layer (exactly experts.0.*
        # for a valid src_cfg, but be robust) before writing the dst_cfg.n_experts copies.
        for k in list(out.keys()):
            if k.startswith(f"{prefix}experts."):
                del out[k]
        for j in range(dst_cfg.n_experts):
            for lin in _EXPERT_LINS:
                out[f"{prefix}experts.{j}.{lin}.weight"] = src_expert[lin].copy()

        out[f"{prefix}router.weight"] = rng.uniform(
            -router_bound, router_bound, size=(dst_cfg.n_experts, d_model)
        ).astype(np.float32)

        for j in range(dst_cfg.n_shared_experts):
            se_prefix = f"{prefix}shared_experts.{j}."
            if j < src_cfg.n_shared_experts:
                continue                              # already in `out`, copied through
            if shared_expert_init == "forbid":
                raise UpcycleError(
                    f"target has shared expert {j} at layer {i} with no source "
                    "counterpart (src n_shared_experts="
                    f"{src_cfg.n_shared_experts}), and shared_expert_init='forbid' "
                    "rejects synthesizing a new one."
                )
            out[f"{se_prefix}gate.weight"] = rng.uniform(
                -gate_up_bound, gate_up_bound, size=(d_ff, d_model)).astype(np.float32)
            out[f"{se_prefix}up.weight"] = rng.uniform(
                -gate_up_bound, gate_up_bound, size=(d_ff, d_model)).astype(np.float32)
            # Exact zeros: down(silu(gate(x)) * up(x)) is identically 0 regardless of
            # gate/up, which is the whole trick (see the docstring above).
            out[f"{se_prefix}down.weight"] = np.zeros((d_model, d_ff), dtype=np.float32)
        # A target with FEWER shared experts than the source (unusual, but not excluded
        # by check_upcycle_compatible) drops the extras rather than carrying dead keys.
        for k in list(out.keys()):
            if k.startswith(f"{prefix}shared_experts."):
                j = int(k[len(f"{prefix}shared_experts."):].split(".", 1)[0])
                if j >= dst_cfg.n_shared_experts:
                    del out[k]

    expected = _expected_keys(dst_cfg)
    got_keys = set(out)
    if got_keys != set(expected):
        missing = sorted(set(expected) - got_keys)
        unexpected = sorted(got_keys - set(expected))
        raise UpcycleError(
            "internal error: upcycle output key set does not match dst_cfg's expected "
            f"layout — missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    bad_shape = sorted(
        k for k, shp in expected.items() if tuple(np.asarray(out[k]).shape) != shp
    )
    if bad_shape:
        raise UpcycleError(
            "internal error: upcycle output has wrong shapes for: "
            f"{[(k, tuple(np.asarray(out[k]).shape), expected[k]) for k in bad_shape[:10]]}"
        )
    return out


def _expected_keys(cfg: MambaConfig) -> Dict[str, Tuple[int, ...]]:
    """Portable mirror of `MLXMambaModel._portable_state_dict()` /
    `CUDAMambaModel._portable_state_dict()`'s key layout (verified byte-identical between
    the two backends), used ONLY as `upcycle_dense_to_moe`'s internal safety-net
    assertion — this module deliberately never imports a backend to build a real model.
    Keep in lockstep with both backends; `tests/test_upcycle.py` cross-checks this
    against a real MLX model's `_portable_state_dict()` keys so a drift here is caught
    immediately rather than silently defeating the assertion it exists to make."""
    d_model, d_inner, H, N = cfg.d_model, cfg.d_inner, cfg.n_heads, cfg.d_state
    dt_rank = cfg.dt_rank_resolved
    out: Dict[str, Tuple[int, ...]] = {"embedding.weight": (cfg.vocab_size, d_model)}
    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        out[f"{p}norm.weight"] = (d_model,)
        if cfg.is_attention_layer(i):
            nah, dh = cfg.n_attn_heads_resolved, cfg.attn_head_dim
            d_attn = nah * dh
            out[f"{p}qkv_proj.weight"] = (3 * d_attn, d_model)
            out[f"{p}o_proj.weight"] = (d_model, d_attn)
        elif cfg.is_moe_layer(i):
            d_ff = cfg.moe_d_ff_resolved
            out[f"{p}router.weight"] = (cfg.n_experts, d_model)
            for j in range(cfg.n_experts):
                out[f"{p}experts.{j}.gate.weight"] = (d_ff, d_model)
                out[f"{p}experts.{j}.up.weight"] = (d_ff, d_model)
                out[f"{p}experts.{j}.down.weight"] = (d_model, d_ff)
            for j in range(cfg.n_shared_experts):
                out[f"{p}shared_experts.{j}.gate.weight"] = (d_ff, d_model)
                out[f"{p}shared_experts.{j}.up.weight"] = (d_ff, d_model)
                out[f"{p}shared_experts.{j}.down.weight"] = (d_model, d_ff)
        else:
            out[f"{p}in_proj.weight"] = (2 * d_inner, d_model)
            out[f"{p}conv.weight"] = (d_inner, cfg.d_conv, 1)   # MLX-canonical (out,k,in/groups)
            out[f"{p}conv.bias"] = (d_inner,)
            out[f"{p}ssm.x_proj.weight"] = (dt_rank + 2 * N, d_inner)
            out[f"{p}ssm.dt_proj.weight"] = (H, dt_rank)
            out[f"{p}ssm.dt_proj.bias"] = (H,)
            out[f"{p}ssm.A_log"] = (H,)
            out[f"{p}ssm.D"] = (H,)
            out[f"{p}out_proj.weight"] = (d_model, d_inner)
    out["norm_f.weight"] = (d_model,)
    if not cfg.tie_embeddings:
        out["lm_head.weight"] = (cfg.vocab_size, d_model)
    return out


def upcycle_manifest(*, src: str, src_sha256: str, seed: int, router_init_scale: float,
                     shared_expert_init: str, src_cfg: MambaConfig,
                     dst_cfg: MambaConfig) -> Dict[str, Any]:
    """The `<out>.upcycle.json` sidecar `scripts/upcycle.py` writes: enough to know
    exactly what produced an upcycled checkpoint (and to re-run it) without re-deriving
    anything from the weights themselves."""
    moe_layers = sorted(i for i in range(dst_cfg.n_layers) if dst_cfg.is_moe_layer(i))
    return {
        "src": str(src),
        "src_sha256": src_sha256,
        "seed": int(seed),
        "router_init_scale": float(router_init_scale),
        "shared_expert_init": shared_expert_init,
        "n_experts_src": src_cfg.n_experts,
        "n_experts_dst": dst_cfg.n_experts,
        "moe_layers": moe_layers,
    }
