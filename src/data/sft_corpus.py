"""Resolve a built SFT corpus directory to trainable records (#306) — the bridge between the
`shared/sft/` builders and `scripts/sft.py`.

`instruct_sft.py` (#95), `reasoning_sft.py` (#96) and `tool_sft.py` (#102) all write into the
`storage.sft_tokenized_dir` layout::

    <root>/sft/tokenized/<tok>-<k>/
        instruct.jsonl   + manifest.json
        reasoning.jsonl  + reasoning-manifest.json
        reasoning-packed/            <- PRETRAINING artifact, no loss mask (see below)
        tool.jsonl       + tool-manifest.json

`scripts/sft.py` historically hard-coded `<data>/train.jsonl` + `<data>/val.jsonl`, so none of
those files had a training driver. This module resolves either layout to what the driver needs:

- **generic** — a directory that already has `train.jsonl` is returned as *paths*, untouched. The
  pre-#306 invocation is bit-for-bit unchanged: no split, no manifest check.
- **shared** — one or more of the `FORMS` files, read, manifest-validated, concatenated,
  length-filtered and split deterministically into train/val record lists.

**`reasoning-packed/` is rejected by name.** It is `shard.pack_atomic` output: a flat token
stream with `.bounds` marking document *starts* only, every document padded to `chunk_align` and
every sequence tail padded with `pad_id`. Nothing in it distinguishes a real final token from
alignment padding, so **no loss mask can be reconstructed** — an "SFT over packed" mode would have
to invent one, and an invented mask is a silently-wrong training signal. Packed is a pretraining
artifact and gets the pretraining driver (`src.data.split --shards` -> `scripts/train.py`); the
rejection prints that recipe.

**Chat-EOS is verified here, not merely recorded.** `chat_template.CHAT_EOS` is the SFT == RL ==
serving invariant (`docs/design/11-post-training.md`); every builder writes `chat_eos` into its
manifest and, before #306, nothing read it back. A manifest whose `chat_eos` disagrees — or which
lacks the key entirely, which is unknown provenance rather than a pass — is refused.

ABOVE THE SEAM — no `mlx`/`torch`. Pure `json`/`pathlib`/`numpy` plus `chat_template`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import chat_template
from .sft_loader import load_sft_records

#: Masked-JSONL forms: form name -> (records file, manifest file). These are the forms
#: `scripts/sft.py` can train, because each record carries its own `loss_mask`.
FORMS: Dict[str, Tuple[str, str]] = {
    "instruct": ("instruct.jsonl", "manifest.json"),
    "reasoning": ("reasoning.jsonl", "reasoning-manifest.json"),
    "tool": ("tool.jsonl", "tool-manifest.json"),
}

#: Pretraining-only forms — named here so the rejection can name them back. Value is the
#: subdirectory the builder writes.
PACKED_FORMS: Dict[str, str] = {"reasoning-packed": "reasoning-packed"}

#: Pseudo-forms accepted on the CLI.
AUTO = "auto"
GENERIC = "generic"

#: The chat template every `shared/sft/` builder renders under.
TEMPLATE = "qwen-chatml"

#: Manifest fields that must agree when more than one form is mixed into one run. A
#: disagreement means the two corpora were built against different tokenizers/templates,
#: and concatenating them would train one of them on the wrong ids.
MIX_FIELDS: Tuple[str, ...] = ("chat_eos", "template", "tokenizer", "model_id", "seq_len")


def packed_recipe(packed_dir) -> str:
    """The two commands that actually train a packed artifact. Printed by every rejection so the
    error carries the fix rather than only the diagnosis."""
    return (f"`reasoning-packed/` carries no loss mask, so it is a PRETRAINING artifact and "
            f"scripts/sft.py cannot train it. Use the pretraining driver:\n"
            f"    python -m src.data.split --shards {packed_dir} --out <split> --val-tokens N\n"
            f"    python scripts/train.py --config <cfg> --data <split>\n"
            f"See docs/design/11-post-training.md (Thinking).")


def looks_packed(d: Path) -> bool:
    """True if `d` is a `shard.pack_atomic` output directory (manifest + part shards) rather than
    a `shared/sft/tokenized/<tok>-<k>` directory."""
    d = Path(d)
    return (d / "manifest.json").exists() and any(d.glob("part-*.bin"))


@dataclass
class SFTCorpus:
    """What `scripts/sft.py` needs to build its two loaders.

    Exactly one of the two shapes is populated:
      * generic  -> `train_path` / `val_path` (files handed straight to `SFTLoader`)
      * shared   -> `train_records` / `val_records` (in-memory lists, no temp files)
    """
    forms: List[str]
    train_records: Optional[List[dict]] = None
    val_records: Optional[List[dict]] = None
    train_path: Optional[Path] = None
    val_path: Optional[Path] = None
    manifests: Dict[str, dict] = field(default_factory=dict)
    n_dropped_overlength: int = 0

    @property
    def is_generic(self) -> bool:
        return self.train_path is not None

    def summary(self) -> str:
        n_tr = len(self.train_records) if self.train_records is not None else "?"
        n_va = len(self.val_records) if self.val_records is not None else "?"
        return (f"forms={'+'.join(self.forms)} train={n_tr} val={n_va} "
                f"dropped_overlength={self.n_dropped_overlength}")


def validate_manifest(path: Path, manifest: dict) -> None:
    """Enforce the cross-cutting chat-template invariants (#306, item 4).

    `chat_eos` MUST be present and equal to `chat_template.CHAT_EOS`, and `template` MUST be
    `qwen-chatml`. A missing key is an error, not a pass: an older manifest is unknown provenance,
    and training under a different stop token degrades the model at serving time in a way nothing
    downstream would catch.
    """
    if "chat_eos" not in manifest:
        raise ValueError(
            f"{path}: manifest has no `chat_eos` key. Expected chat_eos="
            f"{chat_template.CHAT_EOS!r} (chat_template.CHAT_EOS). A corpus of unknown chat "
            f"provenance is not trainable — rebuild it with the current builder.")
    found = manifest["chat_eos"]
    if found != chat_template.CHAT_EOS:
        raise ValueError(
            f"{path}: chat_eos mismatch — expected {chat_template.CHAT_EOS!r} "
            f"(chat_template.CHAT_EOS), found {found!r}. SFT, RL and serving must share one "
            f"chat EOS (docs/design/11-post-training.md); rebuild the corpus.")
    tmpl = manifest.get("template")
    if tmpl != TEMPLATE:
        raise ValueError(
            f"{path}: template mismatch — expected {TEMPLATE!r}, found {tmpl!r}. "
            f"scripts/sft.py trains the Qwen ChatML layout only.")


def _check_mixing(manifests: Dict[str, dict]) -> None:
    """Every mixed form must agree on tokenizer/template/model/seq_len, else the concatenation
    trains one corpus on another's ids."""
    names = list(manifests)
    if len(names) < 2:
        return
    ref = names[0]
    for other in names[1:]:
        for key in MIX_FIELDS:
            a, b = manifests[ref].get(key), manifests[other].get(key)
            if a != b:
                raise ValueError(
                    f"cannot mix SFT forms {ref!r} and {other!r}: `{key}` disagrees "
                    f"({ref}={a!r}, {other}={b!r}). Rebuild both forms with the same "
                    f"tokenizer/template/seq_len, or train them in separate runs.")


def _resolve_form_names(data_dir: Path, forms: Optional[Sequence[str]]) -> List[str]:
    """Normalize the requested `--corpus-form` list to concrete `FORMS` keys, rejecting packed by
    name and autodetecting when asked."""
    requested = [f for f in (forms or [AUTO])]
    for f in requested:
        if f in PACKED_FORMS:
            raise ValueError(
                f"--corpus-form {f}: not an SFT form. " + packed_recipe(data_dir / PACKED_FORMS[f]))
        if f not in FORMS and f not in (AUTO, GENERIC):
            raise ValueError(f"unknown SFT corpus form {f!r} "
                             f"(have {sorted(FORMS)} + {sorted(PACKED_FORMS)} + [{AUTO}, {GENERIC}])")
    if AUTO in requested or GENERIC in requested:
        if len(requested) > 1:
            raise ValueError(f"--corpus-form {AUTO}/{GENERIC} cannot be combined with a "
                             f"specific form (got {requested})")
        if requested == [GENERIC]:
            return [GENERIC]
        # auto: generic wins if train.jsonl is there, else detect the shared-layout files.
        if (data_dir / "train.jsonl").exists():
            return [GENERIC]
        present = [name for name in FORMS if (data_dir / FORMS[name][0]).exists()]
        if not present:
            raise ValueError(
                f"no SFT corpus found in {data_dir}: expected either train.jsonl (generic) or "
                f"one of {[FORMS[n][0] for n in FORMS]} (the shared/sft layout written by "
                f"src.data.{{instruct,reasoning,tool}}_sft). "
                + (packed_recipe(data_dir) if looks_packed(data_dir) else ""))
        return present
    return requested


def _split_indices(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """One seeded permutation; the tail is val. Same seed -> same split, so a run is reproducible
    from the `train=/val=` line in its log."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    order = np.random.default_rng(seed).permutation(n)
    n_val = min(max(1, math.ceil(val_frac * n)), n - 1)
    return order[: n - n_val], order[n - n_val:]


def resolve_sft_corpus(data_dir, forms: Optional[Sequence[str]] = None, *,
                       val_frac: float = 0.05, seed: int = 0,
                       max_len: Optional[int] = None) -> SFTCorpus:
    """Resolve `data_dir` to trainable train/val records (or, in generic mode, to the two paths).

    `forms` is the `--corpus-form` list: `None`/`["auto"]` autodetects, `["generic"]` forces the
    pre-#306 `train.jsonl`/`val.jsonl` behavior, and any subset of `FORMS` reads those files from
    the `shared/sft/tokenized/<tok>-<k>` layout. `reasoning-packed` is rejected by name.

    `max_len` drops (never truncates) records whose `input_ids` are longer, matching the builders:
    truncating a record clips the answer span off the end of the mask, which teaches the model to
    stop mid-answer. Dropped records are counted in `SFTCorpus.n_dropped_overlength`.
    """
    data_dir = Path(data_dir)
    if looks_packed(data_dir):
        raise ValueError(f"{data_dir} is a packed shard directory, not an SFT corpus. "
                         + packed_recipe(data_dir))

    names = _resolve_form_names(data_dir, forms)

    # -- generic: hand back the two paths untouched (pre-#306 behavior, bit for bit) ----------
    if names == [GENERIC]:
        train_path = data_dir / "train.jsonl"
        if not train_path.exists():
            raise ValueError(f"--corpus-form {GENERIC}: {train_path} does not exist "
                             f"(a generic SFT corpus is train.jsonl + val.jsonl from "
                             f"src.data.sft_data).")
        val_path = data_dir / "val.jsonl"
        if val_path.exists():
            return SFTCorpus(forms=[GENERIC], train_path=train_path, val_path=val_path)
        # No held-out file: split train.jsonl rather than evaluate on the training set.
        records = load_sft_records(train_path)
        return _split_to_corpus([GENERIC], records, {}, val_frac, seed, max_len, data_dir)

    # -- shared/sft layout --------------------------------------------------------------------
    manifests: Dict[str, dict] = {}
    records: List[dict] = []
    for name in names:
        rec_file, man_file = FORMS[name]
        rec_path, man_path = data_dir / rec_file, data_dir / man_file
        if not rec_path.exists():
            raise ValueError(
                f"--corpus-form {name}: {rec_path} does not exist. Build it with "
                f"`python -m src.data.{'reasoning' if name == 'reasoning' else name}_sft "
                f"--out-root <root>` (it writes into {data_dir.parent.parent.parent}).")
        if not man_path.exists():
            raise ValueError(
                f"--corpus-form {name}: {rec_path} has no manifest at {man_path}. The manifest "
                f"records the tokenizer and chat_eos this corpus was built with, and #306 "
                f"verifies them before training — an unaccompanied records file is not trainable.")
        manifest = json.loads(man_path.read_text())
        validate_manifest(man_path, manifest)
        manifests[name] = manifest
        records.extend(load_sft_records(rec_path))

    _check_mixing(manifests)
    return _split_to_corpus(names, records, manifests, val_frac, seed, max_len, data_dir)


def _split_to_corpus(names: List[str], records: List[dict], manifests: Dict[str, dict],
                     val_frac: float, seed: int, max_len: Optional[int],
                     data_dir: Path) -> SFTCorpus:
    """Length-filter, fail loudly on empty, then split deterministically."""
    if max_len is not None and max_len < 0:
        raise ValueError(f"--max-len must be >= 0 (0 = keep all), got {max_len}")

    n_raw = len(records)
    if n_raw == 0:
        raise ValueError(f"empty SFT corpus: {data_dir} yielded 0 records for forms "
                         f"{'+'.join(names)}. Nothing to train on.")

    n_dropped = 0
    if max_len:
        kept = [r for r in records if len(r["input_ids"]) <= max_len]
        n_dropped = n_raw - len(kept)
        records = kept
        if not records:
            raise ValueError(
                f"SFT corpus emptied by the over-length filter: all {n_raw} records in "
                f"{data_dir} (forms {'+'.join(names)}) are longer than max_len={max_len}. "
                f"Records are dropped, never truncated (truncating clips the answer span off "
                f"the mask) — raise --max-len, or use a config with a larger seq_len.")

    n = len(records)
    if n < 2:
        raise ValueError(
            f"SFT corpus in {data_dir} (forms {'+'.join(names)}) has {n} record(s) after "
            f"filtering ({n_dropped} dropped over max_len={max_len}); at least 2 are needed to "
            f"hold out a validation record. Build a larger corpus.")

    tr_idx, va_idx = _split_indices(n, val_frac, seed)
    return SFTCorpus(forms=names,
                     train_records=[records[int(i)] for i in tr_idx],
                     val_records=[records[int(i)] for i in va_idx],
                     manifests=manifests,
                     n_dropped_overlength=n_dropped)
