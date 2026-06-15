"""Clean-license DPO preference sources (#77).

Hermetic — no network, no model. The OASST1 ranking pairs, SHP mapping, and on-policy
pair construction are the contract; verified end to end through `build_dpo_records`.
"""

from src.data.dpo_sources import (SOURCE_LICENSES, build_oasst1_prefs, pairs_from_scored,
                                  shp_to_pref)
from src.data.dpo_data import build_dpo_records
from src.data.tokenize import ByteTokenizer


def test_oasst1_prefs_rank_best_vs_worst():
    rows = [
        {"message_id": "p", "parent_id": None, "role": "prompter", "text": "Q?", "lang": "en"},
        {"message_id": "a", "parent_id": "p", "role": "assistant", "text": "best", "rank": 0, "lang": "en"},
        {"message_id": "b", "parent_id": "p", "role": "assistant", "text": "mid", "rank": 1, "lang": "en"},
        {"message_id": "c", "parent_id": "p", "role": "assistant", "text": "worst", "rank": 2, "lang": "en"},
    ]
    recs = list(build_oasst1_prefs(rows))
    assert len(recs) == 1
    r = recs[0]
    assert r["prompt"] == "Q?"
    assert r["chosen"][0]["content"] == "best" and r["rejected"][0]["content"] == "worst"
    assert r["source"] == "oasst1" and r["license"] == "apache-2.0"


def test_oasst1_prefs_needs_two_ranked_children():
    rows = [
        {"message_id": "p", "parent_id": None, "role": "prompter", "text": "Q?", "lang": "en"},
        {"message_id": "a", "parent_id": "p", "role": "assistant", "text": "only", "rank": 0, "lang": "en"},
    ]
    assert list(build_oasst1_prefs(rows)) == []


def test_shp_to_pref_uses_label():
    a = {"history": "ctx", "human_ref_A": "A", "human_ref_B": "B", "labels": 1}
    b = {"history": "ctx", "human_ref_A": "A", "human_ref_B": "B", "labels": 0}
    ra, rb = shp_to_pref(a), shp_to_pref(b)
    assert ra["chosen"][0]["content"] == "A" and ra["rejected"][0]["content"] == "B"
    assert rb["chosen"][0]["content"] == "B" and rb["rejected"][0]["content"] == "A"
    assert shp_to_pref({"history": "", "human_ref_A": "A", "human_ref_B": "B"}) is None


def test_pairs_from_scored_best_vs_worst():
    pref = pairs_from_scored("prompt", [("low", 0.1), ("high", 0.9), ("mid", 0.5)])
    assert pref["chosen"][0]["content"] == "high" and pref["rejected"][0]["content"] == "low"
    assert pref["source"] == "onpolicy" and pref["license"] == "cc0"
    # need >= 2 distinct-scored non-empty candidates
    assert pairs_from_scored("p", [("only", 0.5)]) is None
    assert pairs_from_scored("p", [("a", 0.5), ("b", 0.5)]) is None
    assert pairs_from_scored("p", [("", 0.1), ("x", 0.9)]) is None


def test_prefs_feed_build_dpo_records():
    rows = [
        {"message_id": "p", "parent_id": None, "role": "prompter", "text": "Hello?", "lang": "en"},
        {"message_id": "a", "parent_id": "p", "role": "assistant", "text": "Good answer here", "rank": 0, "lang": "en"},
        {"message_id": "b", "parent_id": "p", "role": "assistant", "text": "Bad answer", "rank": 1, "lang": "en"},
    ]
    prefs = list(build_oasst1_prefs(rows))
    prefs.append(pairs_from_scored("Q?", [("worse one", 0.2), ("better one", 0.8)]))
    tok = ByteTokenizer()
    stats: dict = {}
    out = list(build_dpo_records(prefs, tok, stats=stats))
    assert out and stats["kept"] == len(out)
    r = out[0]
    assert {"chosen_input_ids", "chosen_mask", "rejected_input_ids", "rejected_mask"} <= set(r)


def test_source_licenses_present():
    assert set(SOURCE_LICENSES) >= {"oasst1", "shp", "onpolicy"}
