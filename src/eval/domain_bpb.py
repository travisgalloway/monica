"""Held-out bits-per-byte per domain / language (#221, elevating #192).

ABOVE THE SEAM. Pure numpy + stdlib; touches a model only through `ModelInterface.forward`
via `src/eval/val_loss.py`.

Why BPB and why per domain
--------------------------
BPB is the primary small-model metric for this program precisely because it is
**tokenizer-invariant**: a code BPE that splits `=>` into one token instead of two changes
perplexity and does not change bits-per-byte. One aggregate BPB, though, cannot say whether
a MoE code model is improving on TypeScript while quietly regressing on prose — which is
the mix question M12's corpus design turns on. So this reports BPB per domain, and the
overall figure is **byte-weighted**, matching how BPB itself is defined.

What this measurement is NOT
----------------------------
Packed shards carry no domain/language field. `src/data/shard.py` writes
`manifest.json = {seq_len, dtype, tokenizer, n_documents, n_sequences, n_tokens, shards}`
with `.bin` + `.bounds` sidecars only, and the Swift packer
(`swift/Sources/MonicaTokenizer/Packing.swift`) writes the identical shape; the per-record
`source`/`lang`/`meta` that exist at the cleaned-Parquet stage are dropped by the time
tokens are packed. `src/data/split.py` additionally writes no `n_bytes`, so `val_bpb` is
*silently omitted* on the shard path today.

So this module reads **purpose-built per-domain val sets** (built by
`scripts/build_domain_val_sets.py` from the cleaned Parquet, where the tags still exist).
That is a different measurement from "BPB per domain over the actual training mix" — it is
BPB over a held-out sample *drawn from each domain*, and it is honest about being that. The
alternative, a `.domains` sidecar written next to `.bounds` by both packers, would let the
number be read off the training corpus directly; it is a data-pipeline change requiring a
re-pack and is filed as the follow-up (see `docs/design/13-code-model-moe.md`).

The BLIND rule
--------------
`val_loss.evaluate` omits `val_bpb` when the packed file records no `n_bytes`. In a
per-domain BPB report that omission would read as "this domain is fine", which is the worst
possible failure shape — a check that cannot observe its target must report loudly, never
healthily. So a domain whose `.meta.json` has no `n_bytes` **raises**, naming the path. A
domain whose val file is too small for even one chunk is recorded as `None` (the
`long_context.py` convention), never `0`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..data.loader import PackedLoader
from ..data.pack import packed_n_bytes
from .val_loss import bits_per_byte, evaluate, perplexity


def load_domain_index(path) -> Dict[str, dict]:
    """Read `domains.json` (written by `scripts/build_domain_val_sets.py`).

    Returns `{domain: {packed, n_docs, n_tokens, n_bytes}}`. The file's `config` block is
    not returned — callers that want the build config read the JSON directly.
    """
    path = Path(path)
    blob = json.loads(path.read_text())
    domains = blob.get("domains")
    if not isinstance(domains, dict) or not domains:
        raise ValueError(f"{path}: no 'domains' block — not a domain index, or it is empty")
    out: Dict[str, dict] = {}
    for name, entry in domains.items():
        packed = Path(entry["packed"])
        if not packed.is_absolute():
            packed = (path.parent / packed).resolve()
        out[name] = {**entry, "packed": str(packed)}
    return out


def evaluate_domain_bpb(model, domains: Dict[str, Path], *, batch_size: int = 4,
                        seq_len: int = 512, max_batches: Optional[int] = None,
                        to_numpy=np.asarray) -> dict:
    """Per-domain held-out CE / perplexity / BPB, plus a byte-weighted overall.

    `domains` maps a domain name to its packed `.bin`. Each domain is read with
    `shuffle=False, drop_last=False` so the evaluated token set is a deterministic function
    of `(seq_len, batch_size, max_batches)` — the same file scored twice gives the same
    number, which is the acceptance bar for this whole suite.

    Raises when a domain's packed file records no `n_bytes` (see the module docstring).
    """
    if not domains:
        raise ValueError("evaluate_domain_bpb(): no domains — an empty report is a failure, "
                         "not a perfect score")

    by_domain: Dict[str, Optional[dict]] = {}
    total_ce_nats = 0.0
    total_tokens = 0
    total_bytes = 0.0

    for name in sorted(domains):
        packed = Path(domains[name])
        if not packed.exists():
            raise FileNotFoundError(f"domain {name!r}: packed val file {packed} does not exist")
        n_bytes = packed_n_bytes(packed)
        if not n_bytes:
            raise ValueError(
                f"domain {name!r}: {packed.with_suffix('.meta.json')} records no 'n_bytes', so "
                "BPB cannot be computed for it. val_loss.evaluate() would silently omit "
                "val_bpb, and a missing per-domain BPB reads as 'this domain is fine'. "
                "Rebuild the val set with scripts/build_domain_val_sets.py, which always "
                "writes n_bytes."
            )
        try:
            loader = PackedLoader(packed, seq_len=seq_len, batch_size=batch_size,
                                  shuffle=False, drop_last=False)
        except ValueError:
            # Too small for even one (seq_len + 1) chunk. Explicitly unmeasured, not zero.
            by_domain[name] = None
            continue

        result = evaluate(model, loader, max_batches=max_batches, to_numpy=to_numpy)
        n_chunks = loader.n_chunks
        if max_batches is not None:
            n_chunks = min(n_chunks, max_batches * batch_size)
        n_tokens_eval = int(n_chunks * seq_len)
        ce_total = float(result["val_loss"]) * n_tokens_eval
        eff_bytes = n_tokens_eval * (loader.n_bytes / loader.n_tokens)

        by_domain[name] = {
            "val_loss": float(result["val_loss"]),
            "val_perplexity": float(result["val_perplexity"]),
            "val_bpb": bits_per_byte(ce_total, eff_bytes),
            "n_tokens": n_tokens_eval,
            "n_bytes": float(eff_bytes),
            "corpus_n_tokens": int(loader.n_tokens),
            "corpus_n_bytes": int(loader.n_bytes),
        }
        total_ce_nats += ce_total
        total_tokens += n_tokens_eval
        total_bytes += eff_bytes

    measured = [n for n, v in by_domain.items() if v is not None]
    if not measured:
        raise ValueError(
            "evaluate_domain_bpb(): every domain was too small to score — no BPB was measured"
        )

    mean_ce = total_ce_nats / total_tokens
    return {
        "by_domain": by_domain,
        "overall": {
            "val_bpb": bits_per_byte(total_ce_nats, total_bytes),   # byte-weighted
            "val_loss": float(mean_ce),                             # token-weighted
            "val_perplexity": perplexity(float(mean_ce)),
            "n_tokens": int(total_tokens),
            "n_bytes": float(total_bytes),
        },
        "n_domains": len(by_domain),
        "n_domains_measured": len(measured),
        "unmeasured_domains": sorted(n for n, v in by_domain.items() if v is None),
    }


def format_domain_bpb_table(results: dict) -> str:
    """One line per domain + a byte-weighted overall line."""
    lines = ["Held-out BPB by domain:"]
    for name, agg in results["by_domain"].items():
        if agg is None:
            lines.append(f"  {name:<24} (val set too small to score — unmeasured)")
            continue
        lines.append(f"  {name:<24} bpb={agg['val_bpb']:.4f}  ce={agg['val_loss']:.4f}  "
                     f"ppl={agg['val_perplexity']:.4f}  "
                     f"({agg['n_tokens']} tok, {agg['n_bytes']:.0f} B)")
    o = results["overall"]
    lines.append(f"  {'overall (byte-wtd)':<24} bpb={o['val_bpb']:.4f}  ce={o['val_loss']:.4f}  "
                 f"ppl={o['val_perplexity']:.4f}  ({o['n_tokens']} tok, {o['n_bytes']:.0f} B)")
    if results["unmeasured_domains"]:
        lines.append(f"  unmeasured: {', '.join(results['unmeasured_domains'])}")
    return "\n".join(lines)
