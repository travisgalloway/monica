# `prose_recall` Fixtures (#363)

Synthetic technical specification definitions and unrelated technical prose distractors used to evaluate associative factual recall across token distance horizons (512, 1024, 2048, 4096, 8192, 16384 tokens).

## Purpose & Architecture Context

Monica's MHM spine employs a recurrent Mamba-2/SSD backbone with ~12.5% hybrid attention layers. In recurrent state-space models, fixed-width recurrent state representation suffers associative recall decay across long distractor horizons. Factual recall over specifications (RFCs, PEPs, system design, and distributed systems architecture) validates that hybrid attention layers maintain factual precision without state washout.

## Schema

### `specs.jsonl`
One JSON object per line, canonical format:
- `id`: Unique identifier for the specification probe.
- `domain`: Domain classification (e.g. `rfc`, `pep`, `distributed_systems`, `system_design`, `posix`).
- `statement`: Technical specification invariant or definition.
- `query`: Focused prompt querying the defined entity.
- `target`: Gold target entity token span.
- `candidates`: Closed candidate pool of in-domain near-miss alternatives (including the target).
- `path`: Virtual path conforming to `src.eval.code_suite.load_code_files`.
- `text`: Text content conforming to `src.eval.code_suite.load_code_files`.

### `distractors.jsonl`
One JSON object per line, canonical format:
- `path`: Virtual path.
- `text`: Substantial technical prose essay on an unrelated technical topic (optics, orbital mechanics, metallurgy, acoustics, horology) with zero entity overlap.

## Provenance
Hand-authored synthetic technical prose. Excluded from training data via decontamination blocklist.
