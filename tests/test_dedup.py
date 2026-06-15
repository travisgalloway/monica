"""Cross-source dedup + benchmark decontamination (#73).

Pure numpy + stdlib; no backend, no network. Determinism (fixed MinHash seed) and the
exact/near/decontam semantics are the contract here.
"""

import numpy as np
import pytest

from src.data.corpus import Record
from src.data.dedup import (Decontaminator, DedupStats, MinHasher, MinHashLSH,
                            decontaminate, exact_dedup, jaccard, near_dedup, shingles)


def _recs(texts):
    return [Record(t, "src") for t in texts]


# --- shingling -----------------------------------------------------------------------
def test_shingles_word_ngrams():
    assert shingles("a b c d e f", size=5) == {"a b c d e", "b c d e f"}
    assert shingles("a b", size=5) == {"a b"}          # short doc -> single shingle
    assert shingles("", size=5) == set()


# --- exact dedup ---------------------------------------------------------------------
def test_exact_dedup_collapses_identical():
    st = DedupStats()
    out = list(exact_dedup(_recs(["hello world", "hello world", "different"]), stats=st))
    assert [r.text for r in out] == ["hello world", "different"]
    assert st.dropped_exact == 1


# --- MinHash signatures --------------------------------------------------------------
def test_minhash_signature_deterministic_and_similarity():
    h = MinHasher(num_perm=128, seed=0)
    a = "the quick brown fox jumps over the lazy dog and runs away quickly"
    near = a + " today"                                # near-identical -> high Jaccard
    far = "completely unrelated text about marine biology and deep ocean currents"
    sa1 = h.signature(shingles(a)); sa2 = h.signature(shingles(a))
    assert np.array_equal(sa1, sa2)                    # deterministic
    assert jaccard(sa1, h.signature(shingles(near))) > jaccard(sa1, h.signature(shingles(far)))


def test_minhashlsh_param_validation():
    with pytest.raises(ValueError):
        MinHashLSH(num_perm=128, bands=17)             # 128 % 17 != 0
    with pytest.raises(ValueError):
        MinHashLSH(num_perm=128, bands=0)              # would ZeroDivision
    with pytest.raises(ValueError):
        MinHashLSH(threshold=1.5)                      # out of [0, 1]
    with pytest.raises(ValueError):
        MinHashLSH(shingle_size=0)


def test_minhash_signature_chunking_is_size_invariant():
    # The chunked signature must be independent of chunk size and self-similar = 1.0.
    h = MinHasher(num_perm=64, seed=3)
    shs = shingles(" ".join(f"w{i}" for i in range(600)))   # ~596 distinct 5-gram shingles
    assert np.array_equal(h.signature(shs, chunk=8), h.signature(shs, chunk=4096))
    assert jaccard(h.signature(shs), h.signature(shs)) == 1.0


# --- near dedup ----------------------------------------------------------------------
def test_near_dedup_catches_paraphrase_keeps_distinct():
    base = "the quick brown fox jumps over the lazy dog " * 3
    dup = base + "extra"                               # >threshold overlap
    distinct = "marine biology studies ocean life across the deep sea floor " * 3
    st = DedupStats()
    out = list(near_dedup(_recs([base, dup, distinct]), threshold=0.7, seed=0, stats=st))
    texts = [r.text for r in out]
    assert base in texts and distinct in texts and dup not in texts
    assert st.dropped_near == 1


def test_near_dedup_deterministic():
    recs = _recs(["alpha beta gamma delta epsilon zeta " * 4,
                  "alpha beta gamma delta epsilon zeta extra " * 4,
                  "totally other words here for variety sake indeed " * 4])
    a = [r.text for r in near_dedup(recs, threshold=0.7, seed=1)]
    b = [r.text for r in near_dedup(recs, threshold=0.7, seed=1)]
    assert a == b


# --- decontamination -----------------------------------------------------------------
def test_decontaminate_strips_benchmark_overlap():
    benchmark = "the capital of france is paris and it sits on the seine river today"
    decon = Decontaminator.from_texts([benchmark], ngram_sizes=(13, 7))
    leaked = "trivia: the capital of france is paris and it sits on the seine river today!"
    clean = "an unrelated paragraph about quantum computing and superconducting qubits here"
    st = DedupStats()
    out = list(decontaminate(_recs([leaked, clean]), decon, stats=st))
    assert [r.text for r in out] == [clean]
    assert st.dropped_decontam == 1


def test_decontaminate_7gram_catches_short_overlap():
    decon = Decontaminator.from_texts(["alpha beta gamma delta epsilon zeta eta"], (13, 7))
    assert decon.contaminated("noise alpha beta gamma delta epsilon zeta eta noise")
    assert not decon.contaminated("alpha beta gamma delta different words entirely here")


# --- build_corpus integration --------------------------------------------------------
def test_build_corpus_dedup_minhash_and_stats(tmp_path):
    pytest.importorskip("pyarrow")
    from src.data.corpus import build_corpus, read_shards
    base = "the quick brown fox jumps over the lazy dog " * 3
    recs = _recs([base, base, base + "x", "a wholly different sentence about birds " * 3])
    ds = DedupStats()
    build_corpus(recs, tmp_path / "cleaned", dedup="minhash", near_threshold=0.7,
                 dedup_stats=ds)
    kept = list(read_shards(tmp_path / "cleaned"))
    assert len(kept) == 2                              # 1 exact dup + 1 near dup removed
    assert ds.dropped_exact == 1 and ds.dropped_near == 1
