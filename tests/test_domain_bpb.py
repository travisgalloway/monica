"""Portable tests for per-domain held-out BPB (#221, `src/eval/domain_bpb.py`).

The BLIND rule is the point of most of these: a domain whose packed file records no
`n_bytes` must RAISE, because `val_loss.evaluate` silently omits `val_bpb` there and a
missing per-domain BPB reads exactly like a healthy one.
"""

import json

import numpy as np
import pytest

from src.data.pack import pack_ids
from src.eval.code_suite import StubCausalModel
from src.eval.domain_bpb import (evaluate_domain_bpb, format_domain_bpb_table,
                                 load_domain_index)
from src.eval.val_loss import bits_per_byte

VOCAB = 256


def _pack(tmp_path, name, n_tokens, *, n_bytes=None, seed=0):
    ids = np.random.default_rng(seed).integers(0, VOCAB, size=n_tokens).astype(np.int64)
    path = tmp_path / name / "val.bin"
    pack_ids(ids, path, dtype=np.dtype(np.uint16), n_bytes=n_bytes)
    return path


def _model():
    return StubCausalModel(vocab_size=VOCAB, seed=0)


def test_reports_bpb_per_domain_with_a_byte_weighted_overall(tmp_path):
    domains = {
        "typescript": _pack(tmp_path, "typescript", 4000, n_bytes=16000, seed=1),
        "prose": _pack(tmp_path, "prose", 2000, n_bytes=6000, seed=2),
    }
    res = evaluate_domain_bpb(_model(), domains, batch_size=2, seq_len=128)

    assert set(res["by_domain"]) == {"prose", "typescript"}
    assert res["n_domains_measured"] == 2 and res["unmeasured_domains"] == []
    for agg in res["by_domain"].values():
        assert agg["val_bpb"] > 0 and agg["n_tokens"] > 0

    # The overall BPB is total-nats / (ln2 * total-bytes), i.e. BYTE-weighted — not the mean
    # of the per-domain figures, which would over-weight a small domain.
    total_nats = sum(a["val_loss"] * a["n_tokens"] for a in res["by_domain"].values())
    total_bytes = sum(a["n_bytes"] for a in res["by_domain"].values())
    assert res["overall"]["val_bpb"] == pytest.approx(bits_per_byte(total_nats, total_bytes))
    naive_mean = np.mean([a["val_bpb"] for a in res["by_domain"].values()])
    assert res["overall"]["val_bpb"] != pytest.approx(naive_mean)


def test_bpb_matches_the_ce_conversion(tmp_path):
    """One byte per token here, so BPB must be exactly CE/ln2 — pinning the unit."""
    domains = {"d": _pack(tmp_path, "d", 3000, n_bytes=3000)}
    res = evaluate_domain_bpb(_model(), domains, batch_size=2, seq_len=100)
    agg = res["by_domain"]["d"]
    assert agg["val_bpb"] == pytest.approx(agg["val_loss"] / np.log(2.0), rel=1e-9)


def test_missing_n_bytes_raises_and_names_the_path(tmp_path):
    domains = {"untagged": _pack(tmp_path, "untagged", 3000, n_bytes=None)}
    with pytest.raises(ValueError, match="records no 'n_bytes'"):
        evaluate_domain_bpb(_model(), domains, batch_size=2, seq_len=100)


def test_a_domain_too_small_to_score_is_none_not_zero(tmp_path):
    domains = {
        "big": _pack(tmp_path, "big", 3000, n_bytes=3000),
        "tiny": _pack(tmp_path, "tiny", 8, n_bytes=8),      # fewer than seq_len + 1 tokens
    }
    res = evaluate_domain_bpb(_model(), domains, batch_size=2, seq_len=100)
    assert res["by_domain"]["tiny"] is None
    assert res["unmeasured_domains"] == ["tiny"]
    assert "unmeasured" in format_domain_bpb_table(res)


def test_every_domain_unscorable_raises(tmp_path):
    domains = {"tiny": _pack(tmp_path, "tiny", 8, n_bytes=8)}
    with pytest.raises(ValueError, match="too small to score"):
        evaluate_domain_bpb(_model(), domains, batch_size=2, seq_len=100)


def test_no_domains_raises(tmp_path):
    with pytest.raises(ValueError, match="no domains"):
        evaluate_domain_bpb(_model(), {}, batch_size=2, seq_len=100)


def test_a_missing_packed_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_domain_bpb(_model(), {"gone": tmp_path / "nope/val.bin"},
                            batch_size=2, seq_len=100)


def test_the_same_file_scores_identically_twice(tmp_path):
    domains = {"d": _pack(tmp_path, "d", 3000, n_bytes=9000)}
    a = evaluate_domain_bpb(_model(), domains, batch_size=2, seq_len=100)
    b = evaluate_domain_bpb(_model(), domains, batch_size=2, seq_len=100)
    assert a["by_domain"]["d"] == b["by_domain"]["d"]


def test_max_batches_caps_the_scored_tokens(tmp_path):
    domains = {"d": _pack(tmp_path, "d", 6000, n_bytes=6000)}
    full = evaluate_domain_bpb(_model(), domains, batch_size=2, seq_len=100)
    capped = evaluate_domain_bpb(_model(), domains, batch_size=2, seq_len=100, max_batches=3)
    assert capped["by_domain"]["d"]["n_tokens"] == 3 * 2 * 100
    assert capped["by_domain"]["d"]["n_tokens"] < full["by_domain"]["d"]["n_tokens"]


# --------------------------------------------------------------------------------------- #
# The domains.json index
# --------------------------------------------------------------------------------------- #

def test_load_domain_index_resolves_relative_packed_paths(tmp_path):
    _pack(tmp_path, "ts", 100, n_bytes=100)
    (tmp_path / "domains.json").write_text(json.dumps({
        "config": {}, "domains": {"ts": {"packed": "ts/val.bin", "n_docs": 1,
                                          "n_tokens": 100, "n_bytes": 100}}}))
    index = load_domain_index(tmp_path / "domains.json")
    assert index["ts"]["packed"] == str((tmp_path / "ts/val.bin").resolve())


def test_load_domain_index_rejects_an_empty_index(tmp_path):
    (tmp_path / "domains.json").write_text(json.dumps({"domains": {}}))
    with pytest.raises(ValueError, match="not a domain index"):
        load_domain_index(tmp_path / "domains.json")
