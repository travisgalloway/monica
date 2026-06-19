"""Three-class artifact storage layout (#97) — the single source of truth for where every frozen
artifact lives, so the student layout stays downstream of all of them and #80's R2 readers/writers
point at one canonical prefix scheme (docs/design/08-corpus-pipeline.md, #65).

    poc-distill/      corpus/{cleaned,tokenized/<tok>-<k>}/  teacher-outputs/{topk-logits,hidden-states}/  manifests/
    shared/           sft/{cleaned/<kind>,tokenized/<tok>-<k>}/  rl/{math-verifiable,code-verifiable}/  eval/
    reserve-pretrain/ cleaned/  tokenized/<ver>-<tok>-<k>/  manifests/

The three classes:
- **poc-distill** — the tokenized distillation corpus (#92) + teacher outputs (#94); drives the
  distillation matching only. Invalidated only by a teacher/tokenizer change, never by the student.
- **shared** — the instruct (#95) / reasoning-trace (#96) / tool-use (#102) SFT corpora + the
  verifiable RL sets (#103). Curated once (much of it teacher inference) and reused unchanged by
  both the POC and the production-reserve run.
- **reserve-pretrain** — the full from-scratch corpus (#70/#71), built only after a layout validates.

Rules this module encodes:
- **Cleaned text and RL problems are tokenizer-agnostic and durable** — their helpers take no
  tokenizer (re-tokenize cheaply when the tokenizer/seq_len changes).
- **Every tokenized folder name-pins tokenizer + seq_len** (`tokenized_dir_name`), so the same
  cleaned source can produce several tokenized views side by side without collision.

Portable: pure path joining, no `mlx`/`torch`, no deps. Helpers return `pathlib.Path` for the
local builds today; the same names are valid R2 prefixes, so #80 reuses them through `s3fs`.
"""

from __future__ import annotations

from pathlib import Path

# The three artifact classes (top-level prefixes).
POC_DISTILL = "poc-distill"
SHARED = "shared"
RESERVE_PRETRAIN = "reserve-pretrain"
CLASSES = (POC_DISTILL, SHARED, RESERVE_PRETRAIN)


def class_root(base, cls: str) -> Path:
    """The root of one artifact class under a bucket/local `base` (e.g. `data/poc-distill`)."""
    if cls not in CLASSES:
        raise ValueError(f"unknown storage class {cls!r} (have {CLASSES})")
    return Path(base) / cls


def tokenized_dir_name(tokenizer: str, seq_len: int) -> str:
    """The name-pin for a tokenized folder: `<tokenizer>-<seqlen_k>` (e.g. `qwen25-8k`). This is the
    one place the tokenizer + seq_len naming convention is defined."""
    return f"{tokenizer}-{seq_len // 1024}k"


# --------------------------------------------------------------------------- #
# poc-distill/ — distillation corpus (#92) + teacher outputs (#94)
# --------------------------------------------------------------------------- #
def corpus_cleaned_dir(poc_distill_root) -> Path:
    """Tokenizer-agnostic cleaned distillation text."""
    return Path(poc_distill_root) / "corpus" / "cleaned"


def corpus_tokenized_dir(poc_distill_root, tokenizer: str, seq_len: int) -> Path:
    """Tokenized distillation shards, name-pinned by tokenizer + seq_len."""
    return Path(poc_distill_root) / "corpus" / "tokenized" / tokenized_dir_name(tokenizer, seq_len)


def teacher_outputs_dir(poc_distill_root, kind: str = "topk-logits") -> Path:
    """Precomputed teacher outputs (#94): `topk-logits` or `hidden-states`."""
    return Path(poc_distill_root) / "teacher-outputs" / kind


def manifests_dir(poc_distill_root) -> Path:
    return Path(poc_distill_root) / "manifests"


# --------------------------------------------------------------------------- #
# shared/ — SFT corpora (#95/#96/#102) + verifiable RL sets (#103) + eval
# --------------------------------------------------------------------------- #
def sft_cleaned_dir(shared_root, kind: str) -> Path:
    """Tokenizer-agnostic cleaned SFT rows for one `kind` (instruct / reasoning-traces / tool)."""
    return Path(shared_root) / "sft" / "cleaned" / kind


def sft_tokenized_dir(shared_root, tokenizer: str, seq_len: int) -> Path:
    """Tokenized SFT artifacts, name-pinned by tokenizer + seq_len (shared across SFT kinds)."""
    return Path(shared_root) / "sft" / "tokenized" / tokenized_dir_name(tokenizer, seq_len)


def rl_dir(shared_root, kind: str) -> Path:
    """Tokenizer-agnostic verifiable RL problems for one `kind` (math-verifiable / code-verifiable)."""
    return Path(shared_root) / "rl" / kind


def eval_dir(shared_root) -> Path:
    return Path(shared_root) / "eval"


# --------------------------------------------------------------------------- #
# reserve-pretrain/ — full from-scratch corpus (#70/#71), built post-validation
# --------------------------------------------------------------------------- #
def reserve_cleaned_dir(reserve_root) -> Path:
    return Path(reserve_root) / "cleaned"


def reserve_tokenized_dir(reserve_root, tokenizer: str, seq_len: int, version: str = "v1") -> Path:
    """Tokenized reserve shards, name-pinned by corpus version + tokenizer + seq_len
    (e.g. `tokenized/v1-qwen25-8k`)."""
    return Path(reserve_root) / "tokenized" / f"{version}-{tokenized_dir_name(tokenizer, seq_len)}"


def reserve_manifests_dir(reserve_root) -> Path:
    return Path(reserve_root) / "manifests"
