"""Length curriculum: ramp `seq_len` up (2k -> 4k -> 16k) as a run progresses (#216).

Why. SSMs extend gracefully, and training long *throughout* is far more expensive than
extending late — attention is O(L^2) and the activation footprint scales with L. So the
run spends its early, high-LR steps on short contexts and only pays for long context in
the second half. `batch_size` shrinks inversely so tokens/step stays roughly constant,
which keeps the LR schedule (absolute-step units, see `schedule.py`) meaningful.

**What this does and does not guarantee — read this before reasoning about epochs.**
`PackedLoader` cuts non-overlapping `seq_len + 1` windows out of one flat `train.bin`;
there is no document structure in the packed stream (see the #215 resolution note in
`docs/design/13-code-model-moe.md`). Chunking at a *different* `seq_len` is therefore a
different partition of the same corpus, so a later stage's windows necessarily re-cover
tokens an earlier stage already saw, at a different alignment. The property #216
delivers is **exact stream reproduction across a kill/resume** — not "no token is ever
seen twice". Do not claim the latter.

Portable: pure stdlib math, no numpy, no backend. The spec is a CLI string, not model
config, matching `config/poc.yaml`'s rule that run params are flags.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Stage:
    """One curriculum stage: a shape `(seq_len, batch_size)` held for `steps` steps."""

    index: int
    until_frac: float       # CUMULATIVE fraction of the run ENDING at this stage
    seq_len: int
    batch_size: int
    steps: int

    def tokens_per_step(self, grad_accum: int) -> int:
        return self.batch_size * self.seq_len * grad_accum


@dataclass(frozen=True)
class LengthCurriculum:
    """An ordered, contiguous sequence of stages covering the whole run.

    Stage lengths are in **optimizer steps**, which is what makes a boundary always land
    on a step boundary — every micro-batch inside one optimizer step therefore has the
    same shape, and a step never mixes shapes.
    """

    stages: tuple[Stage, ...]
    grad_accum: int

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("curriculum needs at least one stage")
        if self.grad_accum < 1:
            raise ValueError(f"grad_accum must be >= 1 (got {self.grad_accum})")

    # -- geometry ---------------------------------------------------------------
    @property
    def total_steps(self) -> int:
        return sum(s.steps for s in self.stages)

    @property
    def first_steps(self) -> tuple[int, ...]:
        """Global step at which each stage begins (cumulative; `first_steps[0] == 0`)."""
        out, acc = [], 0
        for s in self.stages:
            out.append(acc)
            acc += s.steps
        return tuple(out)

    def stage_index_at(self, step: int) -> int:
        """Index of the stage owning `step`, clamped to the last stage past the end.

        The loop never asks past `total_steps`, but the stream can sit exactly at the end
        after its final micro-batch, and the last stage deliberately never terminates.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0 (got {step})")
        idx = bisect_right(self.first_steps, step) - 1
        return min(idx, len(self.stages) - 1)

    def stage_at(self, step: int) -> Stage:
        return self.stages[self.stage_index_at(step)]

    def tokens_at(self, step: int) -> int:
        """Tokens consumed by the single optimizer step `step`."""
        return self.stage_at(step).tokens_per_step(self.grad_accum)

    def tokens_before(self, step: int) -> int:
        """Total tokens consumed by steps `[0, step)` — the resume seed for `tokens_seen`."""
        if step < 0:
            raise ValueError(f"step must be >= 0 (got {step})")
        total, seen = 0, 0
        for s in self.stages:
            take = min(s.steps, max(0, step - seen))
            total += take * s.tokens_per_step(self.grad_accum)
            seen += s.steps
        if step > seen:                       # past the end: charge the last stage's rate
            total += (step - seen) * self.stages[-1].tokens_per_step(self.grad_accum)
        return total

    # -- identity ---------------------------------------------------------------
    def fingerprint(self) -> dict:
        """The shape identity a persisted dataloader position depends on.

        `steps` is deliberately EXCLUDED: extending a run with a larger `--total-tokens`
        must stay legal, and the saved position is expressed in counters, not in steps,
        so it remains exact when the budget grows.
        """
        return {"stage_shapes": [[s.seq_len, s.batch_size] for s in self.stages]}

    def describe(self) -> list[str]:
        """One human-readable line per stage, for the startup banner."""
        firsts = self.first_steps
        return [
            f"stage {s.index}: until_frac={s.until_frac:<6g} seq_len={s.seq_len:<6d} "
            f"batch_size={s.batch_size:<4d} steps={s.steps:<8d} "
            f"tokens/step={s.tokens_per_step(self.grad_accum):<10d} first_step={firsts[i]}"
            for i, s in enumerate(self.stages)
        ]

    # -- construction -----------------------------------------------------------
    @classmethod
    def single(cls, seq_len: int, batch_size: int, steps: int,
               grad_accum: int) -> "LengthCurriculum":
        """The degenerate one-stage curriculum — exactly today's fixed-shape behavior.

        `loop.train` synthesizes this when no curriculum is passed, so there is one code
        path and the no-curriculum case stays byte-identical to the pre-#216 loop.
        """
        return cls(stages=(Stage(index=0, until_frac=1.0, seq_len=seq_len,
                                 batch_size=batch_size, steps=steps),),
                   grad_accum=grad_accum)


def parse_curriculum_spec(spec: str) -> list[tuple[float, int, Optional[int]]]:
    """Parse `"until_frac:seq_len[:batch_size],..."` into validated stage triples.

    `until_frac` is CUMULATIVE — the fraction of the run *ending* at that stage — which is
    the wording issue #216 itself uses. That is easy to confuse with "share of the run",
    so it is validated hard: fractions must be strictly increasing and the last must be
    exactly 1.0. Every failure names the offending stage and echoes the whole spec.

        >>> parse_curriculum_spec("0.25:2048,0.5:4096,1.0:16384")
        [(0.25, 2048, None), (0.5, 4096, None), (1.0, 16384, None)]
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"curriculum spec is empty (got {spec!r}); expected "
                         "'until_frac:seq_len[:batch_size],...'")

    out: list[tuple[float, int, Optional[int]]] = []
    for i, raw in enumerate(spec.split(",")):
        field = raw.strip()
        if not field:
            raise ValueError(f"curriculum stage {i} is empty in spec {spec!r}")
        parts = field.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"curriculum stage {i} ({field!r}) has {len(parts)} fields; expected "
                f"'until_frac:seq_len' or 'until_frac:seq_len:batch_size' in spec {spec!r}")
        try:
            frac = float(parts[0])
            seq_len = int(parts[1])
            batch_size = int(parts[2]) if len(parts) == 3 else None
        except ValueError as exc:
            raise ValueError(f"curriculum stage {i} ({field!r}) is malformed in spec "
                             f"{spec!r}: {exc}") from exc

        if not 0.0 < frac <= 1.0:
            raise ValueError(f"curriculum stage {i} ({field!r}): until_frac must be in "
                             f"(0, 1], got {frac} in spec {spec!r}")
        if seq_len < 1:
            raise ValueError(f"curriculum stage {i} ({field!r}): seq_len must be >= 1, "
                             f"got {seq_len} in spec {spec!r}")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"curriculum stage {i} ({field!r}): batch_size must be >= 1, "
                             f"got {batch_size} in spec {spec!r}")
        if out and frac <= out[-1][0]:
            raise ValueError(
                f"curriculum stage {i} ({field!r}): until_frac must strictly increase "
                f"(previous stage ends at {out[-1][0]}); until_frac is CUMULATIVE, not a "
                f"per-stage share. Spec {spec!r}")
        out.append((frac, seq_len, batch_size))

    if abs(out[-1][0] - 1.0) > 1e-9:
        raise ValueError(f"curriculum's last stage must end at until_frac 1.0 (got "
                         f"{out[-1][0]}); until_frac is CUMULATIVE. Spec {spec!r}")
    return out


def build_curriculum(spec: str, *, base_seq_len: int, base_batch_size: int,
                     grad_accum: int, total_tokens: Optional[int] = None,
                     total_steps: Optional[int] = None) -> LengthCurriculum:
    """Turn a spec string into a `LengthCurriculum` with a concrete step allocation.

    Batch derivation (when a stage omits an explicit `batch_size`): hold tokens/step
    roughly constant at the reference `base_batch_size * base_seq_len`, using a **floor**
    rather than a round — a derived stage can then only ever be *cheaper* than the
    reference, never a surprise OOM at 16k. It will not be exactly constant, and that is
    fine; the banner reports the real per-stage figure.

    Step allocation comes from exactly one of `total_tokens` or `total_steps`. Each stage
    gets at least one step, so `total_steps` on a degenerate spec can differ from the
    requested `S` by a step or two — `LengthCurriculum.total_steps` is authoritative and
    the caller prints it.
    """
    if (total_tokens is None) == (total_steps is None):
        raise ValueError("build_curriculum needs exactly one of total_tokens/total_steps")
    if base_seq_len < 1 or base_batch_size < 1:
        raise ValueError(f"base shape must be >= 1 (got seq_len={base_seq_len}, "
                         f"batch_size={base_batch_size})")

    triples = parse_curriculum_spec(spec)
    reference_tokens = base_batch_size * base_seq_len

    stages: list[Stage] = []
    prev_frac = 0.0
    for i, (frac, seq_len, batch_size) in enumerate(triples):
        bs = batch_size if batch_size is not None else max(1, reference_tokens // seq_len)
        if total_tokens is not None:
            share = round(frac * total_tokens) - round(prev_frac * total_tokens)
            steps = max(1, share // (bs * seq_len * grad_accum))
        else:
            steps = max(1, round(frac * total_steps) - round(prev_frac * total_steps))
        stages.append(Stage(index=i, until_frac=frac, seq_len=seq_len,
                            batch_size=bs, steps=int(steps)))
        prev_frac = frac

    return LengthCurriculum(stages=tuple(stages), grad_accum=grad_accum)
