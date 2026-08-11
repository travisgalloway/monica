"""Loss-Free-Balancing policy for MoE routers (portable — never imports a backend).

DeepSeek-V3-style Loss-Free-Balancing (Wang et al., 2024): instead of an auxiliary
load-balancing loss added to the training objective (which fights the primary loss and
needs a hand-tuned weight), a per-expert BIAS is added to the router's *selection*
score only — the gate weight used to combine expert outputs stays the UNBIASED softmax
probability. The bias is nudged after every optimizer step from that step's per-expert
token counts, entirely OUTSIDE the gradient:

    s_i = logit_i + b_i            # selection score (ranks the top-k)
    gate_i = softmax(logit)_i      # UNBIASED — bias never touches the combination weight
    b_i += rate * sign(mean(load) - load_i)     # after the step, from routed-token counts

Under-used experts (load below the layer mean) gain bias and route more; over-used
experts lose bias and route less. No aux loss touches the objective.

One policy object manages ALL MoE layers (`bias` is `[n_layers][n_experts]`), mirroring
the single-object `DynamicLossScaler` precedent (`src/train/loss_scale.py`) and giving a
single place for the numbers to live. This module holds ONLY the number policy — the
backend computes hardware-dependent per-expert load counts (from its own router forward)
and calls `update(loads)` here; keeping the policy above the seam makes it unit-testable
without MLX and ready for a future CUDA MoE router (#214) to reuse unchanged.

Persistence note (#213 D3): the bias itself is NOT checkpointed here. It rides in the
PORTABLE weights (`MLXMambaModel._portable_state_dict`'s `moe_route_bias.*` keys) because
it is routing state that changes the model's function at inference — a model served or
evaluated from `weights.safetensors` must route the way it was trained. The drivers
re-seed this object from the model after a resume (`load_state_dict({"bias":
model.moe_biases()})`), so there is exactly one copy of the numbers. `rate` is a
hyperparameter read from config, not state, so the bias vector is this policy's entire
state.
"""

from __future__ import annotations


class MoEBalancer:
    def __init__(self, n_layers: int, n_experts: int, rate: float = 1e-3):
        self.n_layers = int(n_layers)
        self.n_experts = int(n_experts)
        self.rate = float(rate)
        self.bias: list = [[0.0] * self.n_experts for _ in range(self.n_layers)]

    def update(self, loads: list) -> None:
        """Nudge each layer's per-expert bias from that layer's per-expert token counts.

        `loads` is `[n_layers][n_experts]` (this step's routed-token counts, per MoE
        layer in layer order). Under-used experts (count below the layer mean) gain
        bias; over-used experts lose it — a unit step in `sign(mean - count)`, scaled by
        `rate`. Pure Python; no numpy/backend.

        Invariant to a uniform positive scale on `loads` (only the sign of the deviation
        matters), which is what makes the `grad_checkpoint` double-count harmless — see
        `MoEBlock._moe`'s counting comment (#213 D5).

        Raises on a shape mismatch rather than `zip`-truncating it: a short/long `loads`
        means the model and the policy disagree about the MoE layout, and silently
        skipping a layer would mis-steer routing for the rest of the run.
        """
        self._check_shape(loads, "loads")
        for layer_bias, layer_loads in zip(self.bias, loads):
            if not layer_loads:
                continue
            mean = sum(layer_loads) / len(layer_loads)
            for i, load in enumerate(layer_loads):
                diff = mean - load
                sign = (diff > 0) - (diff < 0)   # -1, 0, or +1 (no numpy dependency)
                layer_bias[i] += self.rate * sign

    def _check_shape(self, rows: list, what: str) -> None:
        """Assert `rows` is `[n_layers][n_experts]`, allowing an EMPTY row (a layer with
        no signal yet, or a router that carries no bias). Shared by `update` and
        `load_state_dict` so both fail loudly and identically."""
        if len(rows) != self.n_layers:
            raise ValueError(
                f"{what} has {len(rows)} layers, expected n_layers={self.n_layers}."
            )
        for i, row in enumerate(rows):
            if len(row) not in (0, self.n_experts):
                raise ValueError(
                    f"{what}[{i}] has {len(row)} entries, expected "
                    f"n_experts={self.n_experts} (or 0 for no signal)."
                )

    def biases(self) -> list:
        """Current per-layer bias vectors, for the backend to push into its routers."""
        return [list(b) for b in self.bias]

    @staticmethod
    def utilization_variance(loads: list) -> list:
        """Per-layer variance of the load distribution, normalized to fractions of that
        layer's total routed tokens — the de-collapse metric (lower = more even
        utilization across experts). A layer with zero total load reports 0.0 (no
        signal yet, not "perfectly balanced")."""
        out = []
        for layer_loads in loads:
            total = sum(layer_loads)
            if total <= 0 or not layer_loads:
                out.append(0.0)
                continue
            fracs = [x / total for x in layer_loads]
            n = len(fracs)
            mean = sum(fracs) / n
            out.append(sum((f - mean) ** 2 for f in fracs) / n)
        return out

    def state_dict(self) -> dict:
        return {"bias": self.biases()}

    def load_state_dict(self, state: dict) -> None:
        """Seed the bias from a snapshot — either this object's own `state_dict()` or
        `{"bias": model.moe_biases()}` read back off a just-resumed checkpoint (D3).

        Defensive like `DynamicLossScaler.load_state_dict`: a missing/empty/None state,
        or one whose `bias` key is absent or null, is a no-op rather than a `KeyError`
        (an old checkpoint, or a resume into a config that only just turned balancing
        on). A PRESENT but wrong-shaped bias is an error, not a no-op — it means the
        checkpoint was trained under a different MoE layout, and adopting it would
        mis-steer every router.
        """
        if not state:
            return
        rows = state.get("bias") or []
        if not rows:
            return
        self._check_shape(rows, "bias")
        self.bias = [list(row) if row else [0.0] * self.n_experts for row in rows]


def moe_routing_metrics(stats: list) -> dict:
    """Flat, JSON-serializable per-step routing metrics from `pop_moe_routing_stats()` (#217).

    `stats` is one dict per MoE layer: `{"load", "entropy", "n_tokens"}`. Emits

      moe_util_var                 max over layers  (UNCHANGED key + meaning)
      moe_util_var_per_layer       list[float]
      moe_router_entropy           mean over layers (nats)
      moe_router_entropy_per_layer list[float]

    Each pair is emitted ONLY when it was actually observed — a layer set with zero total
    routed load reports NO `moe_util_var` (0.0 would read as "perfectly balanced" rather
    than "no signal", the BLIND failure), and layers with `entropy is None` (counting off,
    or k==E with counting off) contribute no entropy key. So an un-instrumented step emits
    exactly what it emits today: nothing.
    """
    out: dict = {}
    loads = [s["load"] for s in stats]
    if any(sum(layer_loads) > 0 for layer_loads in loads):
        var_per_layer = MoEBalancer.utilization_variance(loads)
        out["moe_util_var"] = max(var_per_layer)
        out["moe_util_var_per_layer"] = var_per_layer
    entropies = [s["entropy"] for s in stats]
    observed = [e for e in entropies if e is not None]
    if observed:
        out["moe_router_entropy"] = sum(observed) / len(observed)
        out["moe_router_entropy_per_layer"] = entropies
    return out


def balancer_for_config(cfg) -> "MoEBalancer | None":
    """Map a `MambaConfig` to the `MoEBalancer` its training driver needs, or `None`.

    `None` when MoE is off entirely (`moe_every` unset, or it selects no layer) OR when
    `moe_balance_rate` is unset — the latter is the OFF switch: balancing stays off and
    the router path stays byte-identical to pre-#213 behavior (see `MoEBlock._moe`).
    Single source of truth for the config->balancer wiring, kept here (above the seam)
    so it is unit-testable without a backend — parallels `scaler_for_precision`. See the
    training drivers (`scripts/train.py` et al.) for how it is consumed.
    """
    if cfg.moe_every is None or cfg.n_moe_layers == 0:
        return None
    if cfg.moe_balance_rate is None:
        return None
    return MoEBalancer(cfg.n_moe_layers, cfg.n_experts, cfg.moe_balance_rate)


def attach_balancer(balancer, model) -> None:
    """Connect a `MoEBalancer` to a backend model before the first training step.

    Three things, identical in every driver (`scripts/train.py`, `sft.py`, `dpo.py`,
    `rlvr.py`), so they live here rather than being copied four times:

      1. Adopt whatever bias arrived with the model's weights. After a resume (or an
         SFT/DPO/RLVR init from a balanced checkpoint) the bias rides in the PORTABLE
         weights (#213 D3), so the model — not a separate bundle field — is the single
         source of truth; a fresh model reports empty vectors and keeps the zero init.
      2. Push the resulting bias back into the routers, so the FIRST step already routes
         with it instead of waiting for the first post-step push.
      3. Turn on per-expert load counting (#213 D4), which is off by default.

    A no-op when `balancer` is None (the off switch), so drivers can call it
    unconditionally. `model` is duck-typed (`moe_biases` / `set_moe_biases` /
    `set_moe_load_counting`) — this module still imports no backend.
    """
    if balancer is None:
        return
    loaded = model.moe_biases()
    if loaded and all(row for row in loaded):     # every MoE layer carried a bias
        balancer.load_state_dict({"bias": loaded})
    model.set_moe_biases(balancer.biases())
    model.set_moe_load_counting(True)
