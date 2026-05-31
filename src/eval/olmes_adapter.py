"""Tier-2 evaluation: OLMES / lm-evaluation-harness adapter — STUB (deferred).

OLMES inherits its model abstraction from EleutherAI's lm-evaluation-harness and is
built around HuggingFace-loadable models. Evaluating a custom MLX Mamba requires
implementing the harness's model class (the loglikelihood-style methods), using its
`huggingface.py` as reference. This is its own milestone-sized task, NOT wiring.

Known trap: off-by-one errors in loglikelihood token indexing. Run a small set
(HellaSwag, ARC, PIQA). For a 100M model, absolute scores will be poor — judge the
harness by whether it runs end to end, not by leaderboard position.

Deferred for the POC (success = Tier-1 val perplexity). Implement here when
comparable benchmark numbers are wanted.
"""

from __future__ import annotations

from ..model.interface import ModelInterface


class OlmesMambaAdapter:  # pragma: no cover - deferred
    def __init__(self, model: ModelInterface):
        raise NotImplementedError(
            "OLMES adapter is a deferred, milestone-sized task. Implement the "
            "lm-eval model class (loglikelihood + loglikelihood_rolling) over "
            "ModelInterface.forward; mind loglikelihood token indexing off-by-one."
        )
