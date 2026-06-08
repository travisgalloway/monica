"""Dynamic loss-scale policy for fp16 training (portable — never imports a backend).

fp16 gradients underflow to zero for small values and overflow to inf/nan for large
ones. The standard fix is to scale the loss up by a factor S before backprop (shifting
gradients into fp16's representable range), then unscale the grads by 1/S before the
optimizer step. A static S is fragile; this picks S adaptively:

  * on a non-finite gradient (overflow) — drop the step, multiply S by `backoff`;
  * after `growth_interval` consecutive clean steps — multiply S by `growth_factor`.

This module holds ONLY the number policy. The actual inf/nan detection on the gradient
tensors lives in the backend `train_step` (it needs the hardware array type); the backend
calls `update(overflow=...)` here. Keeping the policy above the seam makes it unit-testable
without MLX. `state_dict`/`load_state_dict` let the scale survive a resume.
"""

from __future__ import annotations


class DynamicLossScaler:
    def __init__(
        self,
        init_scale: float = 2.0 ** 13,
        growth_factor: float = 2.0,
        backoff: float = 0.5,
        growth_interval: int = 2000,
        min_scale: float = 1.0,
    ):
        self.scale = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff = float(backoff)
        self.growth_interval = int(growth_interval)
        self.min_scale = float(min_scale)
        self._good_steps = 0

    def update(self, overflow: bool) -> None:
        """Advance the scale given whether the last step's gradients overflowed."""
        if overflow:
            self.scale = max(self.min_scale, self.scale * self.backoff)
            self._good_steps = 0
            return
        self._good_steps += 1
        if self._good_steps >= self.growth_interval:
            self.scale *= self.growth_factor
            self._good_steps = 0

    def state_dict(self) -> dict:
        return {"scale": self.scale, "good_steps": self._good_steps}

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self.scale = float(state["scale"])
        self._good_steps = int(state.get("good_steps", 0))
