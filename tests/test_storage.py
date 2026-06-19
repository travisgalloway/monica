"""Three-class storage layout (#97): the single source of truth for artifact prefixes.
Pure path logic (no backend); the driver-integration test uses the offline byte fallback.
"""

import pytest

from src.data import storage
from src.data.storage import (CLASSES, POC_DISTILL, RESERVE_PRETRAIN, SHARED, class_root,
                              tokenized_dir_name)


def test_classes_and_class_root():
    assert CLASSES == (POC_DISTILL, SHARED, RESERVE_PRETRAIN)
    assert class_root("data", POC_DISTILL).as_posix() == "data/poc-distill"
    assert class_root("data", SHARED).as_posix() == "data/shared"
    with pytest.raises(ValueError):
        class_root("data", "nope")


def test_tokenized_dir_name_pins_tokenizer_and_seqlen():
    assert tokenized_dir_name("qwen25", 8192) == "qwen25-8k"
    assert tokenized_dir_name("olmo", 1024) == "olmo-1k"


def test_poc_distill_layout():
    r = class_root("data", POC_DISTILL)
    assert storage.corpus_cleaned_dir(r).as_posix() == "data/poc-distill/corpus/cleaned"
    assert storage.corpus_tokenized_dir(r, "qwen25", 8192).as_posix() == \
        "data/poc-distill/corpus/tokenized/qwen25-8k"
    assert storage.teacher_outputs_dir(r).as_posix() == \
        "data/poc-distill/teacher-outputs/topk-logits"
    assert storage.teacher_outputs_dir(r, "hidden-states").as_posix() == \
        "data/poc-distill/teacher-outputs/hidden-states"
    assert storage.manifests_dir(r).as_posix() == "data/poc-distill/manifests"


def test_shared_layout():
    r = class_root("data", SHARED)
    assert storage.sft_cleaned_dir(r, "instruct").as_posix() == "data/shared/sft/cleaned/instruct"
    assert storage.sft_cleaned_dir(r, "reasoning-traces").as_posix() == \
        "data/shared/sft/cleaned/reasoning-traces"
    assert storage.sft_tokenized_dir(r, "qwen25", 8192).as_posix() == \
        "data/shared/sft/tokenized/qwen25-8k"
    assert storage.rl_dir(r, "math-verifiable").as_posix() == "data/shared/rl/math-verifiable"
    assert storage.eval_dir(r).as_posix() == "data/shared/eval"


def test_reserve_pretrain_layout():
    r = class_root("data", RESERVE_PRETRAIN)
    assert storage.reserve_cleaned_dir(r).as_posix() == "data/reserve-pretrain/cleaned"
    assert storage.reserve_tokenized_dir(r, "qwen25", 8192).as_posix() == \
        "data/reserve-pretrain/tokenized/v1-qwen25-8k"
    assert storage.reserve_manifests_dir(r).as_posix() == "data/reserve-pretrain/manifests"


def test_cleaned_and_rl_are_tokenizer_free():
    # The tokenizer-agnostic rule: cleaned text + RL problem paths never embed a tokenizer name.
    r_pd, r_sh = class_root("d", POC_DISTILL), class_root("d", SHARED)
    for p in (storage.corpus_cleaned_dir(r_pd), storage.sft_cleaned_dir(r_sh, "instruct"),
              storage.rl_dir(r_sh, "code-verifiable"), storage.reserve_cleaned_dir(
                  class_root("d", RESERVE_PRETRAIN))):
        assert "qwen25" not in p.as_posix() and "-8k" not in p.as_posix()
    # ...while tokenized folders always name-pin it.
    assert "qwen25-8k" in storage.sft_tokenized_dir(r_sh, "qwen25", 8192).as_posix()


def test_drivers_write_under_the_right_class_prefixes(tmp_path):
    # End-to-end: one base, the drivers land artifacts under base/poc-distill and base/shared.
    from src.data.corpus import ingest_dummy
    from src.data.distill_corpus import build_distill_corpus
    from src.data.instruct_sft import build_instruct_sft
    from src.data.reasoning_sft import build_reasoning_sft
    from src.data.sft_sources import handauthored_records
    from src.data.reasoning_traces import handauthored_trace_records

    pytest.importorskip("pyarrow")
    build_distill_corpus(ingest_dummy(300, seed=3), class_root(tmp_path, POC_DISTILL),
                         tokenizer="qwen25", byte_fallback=True, seq_len=1024)
    build_instruct_sft(handauthored_records(), class_root(tmp_path, SHARED),
                       tokenizer="qwen25", byte_fallback=True, seq_len=1024)
    build_reasoning_sft(handauthored_trace_records(), class_root(tmp_path, SHARED),
                        tokenizer="qwen25", byte_fallback=True, seq_len=1024, chunk_align=64)

    assert (tmp_path / "poc-distill" / "corpus" / "tokenized" / "qwen25-1k" / "manifest.json").exists()
    assert (tmp_path / "shared" / "sft" / "tokenized" / "qwen25-1k" / "instruct.jsonl").exists()
    assert (tmp_path / "shared" / "sft" / "tokenized" / "qwen25-1k" / "reasoning.jsonl").exists()
    # The two SFT kinds share the tokenized prefix but split cleaned by kind.
    assert (tmp_path / "shared" / "sft" / "cleaned" / "instruct" / "records.jsonl").exists()
    assert (tmp_path / "shared" / "sft" / "cleaned" / "reasoning-traces" / "records.jsonl").exists()
