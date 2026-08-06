"""The micro-batch stream: stage-aware, with an EXPLICIT, persistable position (#216).

Why this replaces a bare generator. Pre-#216 the data position on resume was
*reconstructed* from `(seed, start_step * grad_accum, per_epoch)` where
`per_epoch = len(loader)` — valid only while `seq_len` (and hence `stride`, `n_chunks`,
`len(loader)`) is fixed for the whole run. A length curriculum changes `seq_len` mid-run,
so the reconstruction silently stops being valid: the run would resume, resample
already-seen data, and get its epoch accounting quietly wrong. On an interruptible pod
that failure looks exactly like success. So the position becomes explicit state, saved
inside the checkpoint slot and restored verbatim.

**Every failure path here raises.** None degrade to "start a fresh epoch" — a silent
resample is the precise bug this module exists to kill.

Portable: no numpy import needed here, no backend. Works with any loader exposing the
duck-typed contract `__len__`, `epoch(reseed, skip_batches)`, `.seq_len`, `.batch_size`
(and optionally `.rng`) — i.e. `PackedLoader`, `SFTLoader`, and `DPOLoader` alike.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from .checkpoint import _jsonable
from .curriculum import LengthCurriculum


STATE_VERSION = 1

# Loader factory: (seq_len, batch_size) -> a loader with the duck-typed contract above.
LoaderFactory = Callable[[int, int], Any]


class MicroBatchStream:
    """Infinite stream of `(inputs, targets)` that walks a `LengthCurriculum`.

    Position is five counters, and they are the whole of the persisted state:

      * ``stage_idx``          — which curriculum stage is active
      * ``micro_in_stage``     — micro-batches consumed inside that stage
      * ``global_micro``       — micro-batches consumed over the whole run
      * ``epoch_idx``          — **global** epoch counter, NOT per-stage
      * ``batches_into_epoch`` — position inside the current epoch

    `epoch_idx` being global is load-bearing: the per-epoch shuffle is reseeded with
    `seed + epoch_idx`, so a global counter makes the single-stage case bit-identical to
    the pre-#216 `_micro_batch_stream`. That backward-compat property is what the legacy
    resume test and the M4 smoke gate depend on, and `test_stream.py` pins it.

    The **last stage never terminates** — it keeps cycling epochs. `train()`'s
    `while step < total_steps` is the only stop condition, which is also what keeps the
    no-curriculum shim unbounded exactly as before.
    """

    def __init__(self, curriculum: LengthCurriculum, loader_factory: LoaderFactory,
                 seed: int, *, extra_fingerprint: Optional[dict] = None):
        self.curriculum = curriculum
        self.loader_factory = loader_factory
        self.seed = int(seed)
        self.extra_fingerprint = dict(extra_fingerprint or {})

        self.stage_idx = 0
        self.micro_in_stage = 0
        self.global_micro = 0
        self.epoch_idx = 0
        self.batches_into_epoch = 0

        self._loaders: dict[int, Any] = {}       # memoized per stage (memmap => cheap)
        self._it = None                          # live, PRIMED epoch iterator
        self._pending = None                     # the batch priming pulled off it

    # -- active stage ------------------------------------------------------------
    @property
    def stage(self):
        return self.curriculum.stages[self.stage_idx]

    @property
    def seq_len(self) -> int:
        return self.stage.seq_len

    @property
    def batch_size(self) -> int:
        return self.stage.batch_size

    @property
    def tokens_per_step(self) -> int:
        return self.stage.tokens_per_step(self.curriculum.grad_accum)

    @property
    def _stage_micro(self) -> int:
        """Micro-batches in the active stage. Stage lengths are in OPTIMIZER STEPS, so a
        boundary always lands on a step boundary — a single step never mixes shapes."""
        return self.stage.steps * self.curriculum.grad_accum

    def loader(self, stage_idx: Optional[int] = None):
        """The loader for a stage, built once and memoized (`open_packed` is a memmap)."""
        i = self.stage_idx if stage_idx is None else stage_idx
        if i not in self._loaders:
            st = self.curriculum.stages[i]
            loader = self.loader_factory(st.seq_len, st.batch_size)
            if len(loader) <= 0:
                # Fail here rather than yielding nothing forever: at long seq_len a small
                # corpus can drop below one batch per epoch (n_chunks < batch_size).
                raise ValueError(
                    f"curriculum stage {i} (seq_len={st.seq_len}, "
                    f"batch_size={st.batch_size}) yields no batches per epoch — the "
                    "corpus is too small for this stage's shape")
            self._loaders[i] = loader
        return self._loaders[i]

    # -- iteration ---------------------------------------------------------------
    def __iter__(self):
        return self

    def _open_epoch(self) -> None:
        """Open the current epoch's iterator **and prime it**.

        Priming is not an optimization. `loader.epoch(...)` is a generator, so its
        `reseed` + shuffle only execute on the first `next()`. Without priming, `.rng`
        would still hold the *previous* epoch's state and the restore tripwire below
        would compare the wrong thing — a tripwire that fires spuriously (or passes
        vacuously) is worse than none.
        """
        loader = self.loader()
        self._it = iter(loader.epoch(reseed=self.seed + self.epoch_idx,
                                     skip_batches=self.batches_into_epoch))
        self._pending = next(self._it, None)

    def _advance_stage(self) -> None:
        """Cross into the next stage: abandon the partial epoch, start a fresh one."""
        self.stage_idx += 1
        self.micro_in_stage = 0
        self.epoch_idx += 1              # global — keeps the reseed sequence unique
        self.batches_into_epoch = 0
        self._it = None
        self._pending = None

    def __next__(self):
        batch = None
        for _ in range(2):               # at most: finish this epoch, then a fresh one
            if self._it is None:
                self._open_epoch()
            if self._pending is not None:
                batch, self._pending = self._pending, None
                break
            nxt = next(self._it, None)
            if nxt is not None:
                batch = nxt
                break
            self.epoch_idx += 1          # epoch exhausted — roll to the next one
            self.batches_into_epoch = 0
            self._it = None
        if batch is None:
            raise ValueError(
                f"curriculum stage {self.stage_idx} produced no batch for a fresh epoch "
                "— the loader's epoch() is empty despite len(loader) >= 1")

        self.micro_in_stage += 1
        self.global_micro += 1
        self.batches_into_epoch += 1

        # Normalize the boundary EAGERLY so a checkpoint taken exactly ON a stage
        # boundary names the new stage at micro_in_stage 0 with a fresh epoch, rather
        # than the exhausted tail of the old one. Nothing is read here — the new stage's
        # epoch is opened lazily on the next call.
        if (self.micro_in_stage >= self._stage_micro
                and self.stage_idx < len(self.curriculum.stages) - 1):
            self._advance_stage()
        return batch

    def seek_micro(self, n: int) -> None:
        """Fast-forward to the position after `n` micro-batches WITHOUT reading them.

        The legacy reconstruction path, kept for pre-#216 resume bundles and for
        `_micro_batch_stream`'s `start_micro`. Only valid for a single-stage curriculum —
        that is the only shape for which the old step->position mapping was ever correct,
        and it is exactly the shape a pre-#216 bundle was written by.
        """
        if n < 0:
            raise ValueError(f"seek_micro needs n >= 0 (got {n})")
        if n and len(self.curriculum.stages) > 1:
            raise ValueError(
                "seek_micro cannot reconstruct a position across a multi-stage "
                "curriculum — the step->position mapping is only valid at a fixed "
                "seq_len. Resume from an explicit data_state, or pass "
                "--ignore-data-state to restart the data stream from the top.")
        per_epoch = len(self.loader())
        self.epoch_idx, self.batches_into_epoch = divmod(n, per_epoch)
        self.micro_in_stage = n
        self.global_micro = n
        self._it = None                  # reopened lazily; skipped chunks are never read
        self._pending = None

    def seek_step(self, step: int) -> None:
        """Reconstruct the position at optimizer `step`, generalized across stages.

        The fallback when no explicit `data_state` is available: a pre-#216 resume
        bundle, or an operator who passed `--ignore-data-state`. It replays the counter
        arithmetic the live stream would have performed — including the extra `epoch_idx`
        bump each stage boundary contributes — using only `len(loader)` per stage, so it
        reads no chunk. Deterministic, but it depends on every stage's `steps`, which is
        exactly the fragility that made explicit state necessary; prefer `data_state`.
        """
        if step < 0:
            raise ValueError(f"seek_step needs step >= 0 (got {step})")
        ga = self.curriculum.grad_accum
        stages = self.curriculum.stages
        epoch_idx, remaining = 0, step
        for i, st in enumerate(stages):
            per_epoch = len(self.loader(i))
            last = i == len(stages) - 1
            take = remaining if last else min(st.steps, remaining)
            micro = take * ga
            # Epochs roll on DETECTED exhaustion, i.e. after per_epoch+1, 2*per_epoch+1,
            # ... consumptions — hence (micro - 1) // per_epoch, not micro // per_epoch.
            rolls = (micro - 1) // per_epoch if micro else 0
            # `remaining == st.steps` on a non-final stage falls through: the live stream
            # normalizes that boundary eagerly onto stage i+1 at micro_in_stage 0.
            if last or remaining < st.steps:
                self.stage_idx = i
                self.micro_in_stage = micro
                self.global_micro = step * ga
                self.epoch_idx = epoch_idx + rolls
                self.batches_into_epoch = micro - rolls * per_epoch if micro else 0
                self._it = None
                self._pending = None
                return
            epoch_idx += rolls + 1       # +1: `_advance_stage` bumps at the boundary
            remaining -= st.steps

    # -- persistence -------------------------------------------------------------
    def fingerprint(self) -> dict:
        """The identity a saved position depends on. A change here invalidates it."""
        return {"seed": self.seed,
                "grad_accum": self.curriculum.grad_accum,
                **self.curriculum.fingerprint(),
                **self.extra_fingerprint}

    def _rng_state(self) -> Any:
        """The live epoch's loader RNG state, or None when there is no epoch in flight.

        Only meaningful mid-epoch: with `_it is None` (a fresh stream, or the instant
        after a stage boundary) the loader still holds its *constructor* seed, which
        would not reproduce on restore. Returning None there makes the tripwire skip
        rather than fire falsely. `getattr` because `FakeLoader`-style test loaders have
        no `.rng` at all.
        """
        if self._it is None:
            return None
        rng = getattr(self._loaders.get(self.stage_idx), "rng", None)
        return rng.bit_generator.state if rng is not None else None

    def state_dict(self) -> dict:
        return {
            "version": STATE_VERSION,
            "stage_idx": self.stage_idx,
            "micro_in_stage": self.micro_in_stage,
            "global_micro": self.global_micro,
            "epoch_idx": self.epoch_idx,
            "batches_into_epoch": self.batches_into_epoch,
            "rng_state": _jsonable(self._rng_state()),
            # Diagnostics only — recorded for a human reading a bundle, never validated.
            "stages_at_save": [[s.seq_len, s.batch_size, s.steps]
                               for s in self.curriculum.stages],
            "fingerprint": self.fingerprint(),
        }

    def load_state_dict(self, state: dict, *, strict: bool = True) -> None:
        """Restore an exact position. Raises on any mismatch — never resumes approximately.

        `strict=False` skips the fingerprint check; the version check always applies,
        because a state we cannot parse is not a state we can approximate.
        """
        if not isinstance(state, dict):
            raise ValueError(f"data_state must be a dict, got {type(state).__name__}")
        version = state.get("version")
        if version != STATE_VERSION:
            raise ValueError(
                f"dataloader state version {version!r} != {STATE_VERSION} — this bundle "
                "was written by a different version of src/train/stream.py. Pass "
                "--ignore-data-state to discard it and restart the data stream.")

        if strict:
            want, got = self.fingerprint(), state.get("fingerprint")
            if got != want:
                raise ValueError(
                    "dataloader state does not match this run's configuration, so its "
                    "saved position is meaningless — refusing to resume with a silently "
                    "wrong data stream.\n"
                    f"  saved:   {json.dumps(got, sort_keys=True)}\n"
                    f"  current: {json.dumps(want, sort_keys=True)}\n"
                    "Restore the original --curriculum/--batch-size/--grad-accum/--seed "
                    "and data path, or pass --ignore-data-state to discard the saved "
                    "position and restart the data stream from the top.")

        n_stages = len(self.curriculum.stages)
        stage_idx = int(state["stage_idx"])
        if not 0 <= stage_idx < n_stages:
            raise ValueError(f"saved stage_idx {stage_idx} is outside this curriculum's "
                             f"{n_stages} stages")

        self.stage_idx = stage_idx
        self.micro_in_stage = int(state["micro_in_stage"])
        self.global_micro = int(state["global_micro"])
        self.epoch_idx = int(state["epoch_idx"])
        self.batches_into_epoch = int(state["batches_into_epoch"])
        self._it = None                  # reopened lazily at the restored position
        self._pending = None

        # Tripwire: reopen the stage's epoch and check the shuffle RNG lands exactly
        # where it did at save time. Coarse (the rng only advances at epoch start), but
        # it catches a wrong seed/epoch_idx pairing and a corpus whose chunk count moved
        # under us — the silent failure modes this module exists to prevent. Comparison
        # only, never fed back into numpy, which is what lets `_jsonable` skip an inverse.
        saved_rng = state.get("rng_state")
        if saved_rng is not None:
            self._open_epoch()           # constructs the loader, reseeds, shuffles
            current = json.loads(json.dumps(_jsonable(self._rng_state())))
            if current != json.loads(json.dumps(saved_rng)):
                raise ValueError(
                    "dataloader RNG tripwire failed after restoring the saved position: "
                    f"seed={self.seed} epoch_idx={self.epoch_idx} reproduces a different "
                    "shuffle than the one that was checkpointed. The resumed data stream "
                    "would NOT match the interrupted run. Pass --ignore-data-state to "
                    "discard the saved position and restart the data stream.")
