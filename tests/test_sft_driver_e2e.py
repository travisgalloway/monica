"""End-to-end proof that the `shared/sft/` builders have a real training driver (#306).

This is the anti-thin-wrapper test. It is not enough that `scripts/sft.py` *imports* the resolver:
the builders' output must reach a real optimizer step and move the loss, and the response mask
must still be the response mask by the time it gets there.

Everything runs at toy scale (`config/toy.yaml`: d_model 64, 2 layers, vocab 256, fp32) over the
checked-in handauthored sources under `ByteTokenizer`, so the whole file is a couple of seconds on
the macOS CI job. MLX-gated — the optimizer step lives below the seam.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.optimizers as optim

from src.data import chat_template
from src.data.reasoning_sft import build_reasoning_sft
from src.data.reasoning_traces import handauthored_trace_records
from src.data.sft_corpus import resolve_sft_corpus
from src.data.sft_loader import SFTLoader, load_sft_records
from src.data.split import split_shards
from src.data.storage import tokenized_dir_name
from src.data.tokenize import ByteTokenizer
from src.data.tool_sft import build_tool_sft
from src.data.tool_sources import (build_abstention_messages, build_tool_messages,
                                   handauthored_tool_records)
from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel
from src.model.mlx_train_step import make_sft_train_step, make_train_step
from src.train.checkpoint import CheckpointStore, load_weights_dict

REPO = Path(__file__).resolve().parents[1]
TOY_CFG = str(REPO / "config" / "toy.yaml")
SEQ_LEN = 1024               # corpus packing length; toy.yaml's seq_len (128) is the loop's
TOKENIZER = "qwen25"         # name-pin only — --byte-fallback supplies the actual tokenizer
MAX_LEN = 1024               # keep every handauthored record (toy seq_len would drop them all)


# --------------------------------------------------------------------------- #
# Fixtures: build the corpora once, and a base checkpoint for --init
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """The three artifacts under test: tool.jsonl, reasoning.jsonl and reasoning-packed/."""
    root = tmp_path_factory.mktemp("shared")
    tool_manifest = build_tool_sft(handauthored_tool_records(), root, tokenizer=TOKENIZER,
                                   byte_fallback=True, seq_len=SEQ_LEN, max_seq_len=SEQ_LEN)
    reasoning_manifest = build_reasoning_sft(handauthored_trace_records(), root,
                                             tokenizer=TOKENIZER, byte_fallback=True,
                                             seq_len=SEQ_LEN, chunk_align=64)
    return {"root": root,
            "tok_dir": root / "sft" / "tokenized" / tokenized_dir_name(TOKENIZER, SEQ_LEN),
            "tool": tool_manifest, "reasoning": reasoning_manifest}


@pytest.fixture(scope="module")
def base_weights(tmp_path_factory):
    """A pretrained-base stand-in — `scripts/sft.py --init` wants portable weights."""
    path = tmp_path_factory.mktemp("base") / "base.safetensors"
    mx.random.seed(0)
    MLXMambaModel(load_config(TOY_CFG)).save(str(path))
    return path


def _sft_driver():
    """`scripts/` is not a package; load the real driver module by path (the same trick
    tests/test_build_domain_val_sets.py uses) so `main()` runs in-process."""
    spec = importlib.util.spec_from_file_location("sft_driver", REPO / "scripts" / "sft.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def _run_driver(monkeypatch, *argv):
    driver = _sft_driver()
    monkeypatch.setattr(sys, "argv", ["sft.py", *[str(a) for a in argv]])
    driver.main()


def _metrics(out: Path):
    return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()
            if line.strip()]


def _train_one_form(monkeypatch, corpus, base_weights, out: Path, form: str, steps: int = 20):
    _run_driver(monkeypatch,
                "--config", TOY_CFG, "--data", corpus["tok_dir"], "--corpus-form", form,
                "--max-len", MAX_LEN, "--init", base_weights, "--out", out,
                "--total-steps", steps, "--batch-size", 2, "--grad-accum", 1,
                "--base-lr", 1e-3, "--log-every", 1, "--eval-every", steps,
                "--ckpt-every", steps, "--seed", 0)
    return _metrics(out)


def _assert_trained(rows, out: Path, steps: int):
    """The run completed, every logged loss is finite, and the loss actually fell."""
    losses = [r["loss"] for r in rows if "loss" in r]
    assert len(losses) == steps, f"expected {steps} logged steps, got {len(losses)}"
    assert all(np.isfinite(x) for x in losses), losses
    assert losses[-1] < losses[0], f"loss did not fall: {losses[0]} -> {losses[-1]}"
    assert (out / "weights.safetensors").exists()

    # <out>/resume reloads to the same step AND the same weights.
    cfg = load_config(TOY_CFG)
    reloaded = MLXMambaModel(cfg)
    opt = optim.AdamW(learning_rate=0.0)
    from src.model.backend import get_backend
    backend = get_backend("mlx")
    meta = CheckpointStore(str(out / "resume")).load(
        weights_deserializer=lambda p: reloaded.load(p),
        optimizer_deserializer=lambda p: backend.load_optimizer(opt, p))
    assert int(meta["step"]) == steps
    final = load_weights_dict(str(out / "weights.safetensors"))
    got = reloaded._portable_state_dict()
    assert set(got) == set(final)
    for k, v in final.items():
        np.testing.assert_allclose(np.asarray(got[k]), np.asarray(v), rtol=0, atol=0,
                                   err_msg=f"resume weights differ at {k}")


# --------------------------------------------------------------------------- #
# 1/2. the two masked forms train through scripts/sft.py, no manual conversion
# --------------------------------------------------------------------------- #

def test_tool_form_trains(monkeypatch, tmp_path, corpus, base_weights):
    out = tmp_path / "run-tool"
    rows = _train_one_form(monkeypatch, corpus, base_weights, out, "tool")
    _assert_trained(rows, out, 20)


def test_reasoning_form_trains(monkeypatch, tmp_path, corpus, base_weights):
    out = tmp_path / "run-reasoning"
    rows = _train_one_form(monkeypatch, corpus, base_weights, out, "reasoning")
    _assert_trained(rows, out, 20)


def test_mixed_forms_train(monkeypatch, tmp_path, corpus, base_weights):
    """`--corpus-form reasoning tool` — the mixing branch, end to end."""
    out = tmp_path / "run-mixed"
    _run_driver(monkeypatch,
                "--config", TOY_CFG, "--data", corpus["tok_dir"],
                "--corpus-form", "reasoning", "tool", "--max-len", MAX_LEN,
                "--init", base_weights, "--out", out, "--total-steps", 10,
                "--batch-size", 2, "--grad-accum", 1, "--base-lr", 1e-3,
                "--log-every", 1, "--eval-every", 10, "--ckpt-every", 10)
    losses = [r["loss"] for r in _metrics(out)]
    assert len(losses) == 10 and all(np.isfinite(x) for x in losses)
    assert losses[-1] < losses[0]


def test_packed_form_rejected_by_the_driver(monkeypatch, tmp_path, corpus, base_weights):
    """The driver itself refuses `reasoning-packed` and prints the pretraining recipe."""
    with pytest.raises(ValueError) as exc:
        _run_driver(monkeypatch, "--config", TOY_CFG, "--data", corpus["tok_dir"],
                    "--corpus-form", "reasoning-packed", "--init", base_weights,
                    "--out", tmp_path / "never")
    msg = str(exc.value)
    assert "reasoning-packed" in msg and "split --shards" in msg and "scripts/train.py" in msg
    assert not (tmp_path / "never").exists()          # nothing was started


# --------------------------------------------------------------------------- #
# 2b. the packed form trains via the driver that IS correct for it
# --------------------------------------------------------------------------- #

def test_packed_form_trains_via_pretrain_driver(tmp_path, corpus):
    """`reasoning-packed/` carries no loss mask, so its driver is
    `src.data.split --shards` -> `PackedLoader` -> `make_train_step`. One real update."""
    from src.data.loader import PackedLoader

    packed = corpus["tok_dir"] / "reasoning-packed"
    split = tmp_path / "reasoning-split"
    train_bin, val_bin = split_shards(packed, split, val_tokens=128)
    assert train_bin.exists() and val_bin.exists()

    cfg = load_config(TOY_CFG)
    loader = PackedLoader(train_bin, cfg.seq_len, 2, seed=0, vocab_size=cfg.vocab_size)
    assert len(loader) >= 1

    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    step = make_train_step(model, optim.AdamW(learning_rate=1e-3), grad_clip=1.0)
    before = np.asarray(model._portable_state_dict()["embedding.weight"]).copy()
    out = step(model, [next(loader.epoch())], 1e-3)
    assert np.isfinite(out["loss"]) and np.isfinite(out["grad_norm"])
    after = np.asarray(model._portable_state_dict()["embedding.weight"])
    assert not np.array_equal(before, after)          # the update actually landed


# --------------------------------------------------------------------------- #
# 3. the response mask is still the response mask at the training step
# --------------------------------------------------------------------------- #

def test_response_mask_through_driver(corpus):
    """A batch drawn from the resolved loader trains the assistant span (through its trailing
    `<|im_end|>`) and nothing else — cross-checked against `chat_template.response_spans`
    recomputed from the cleaned rows — and the step's loss provably depends on that mask.
    """
    cfg = load_config(TOY_CFG)
    resolved = resolve_sft_corpus(corpus["tok_dir"], ["tool"], max_len=MAX_LEN, val_frac=0.34)
    records = resolved.train_records + resolved.val_records
    loader = SFTLoader(corpus["tok_dir"], cfg.seq_len, len(records), shuffle=False,
                       drop_last=False, vocab_size=cfg.vocab_size, records=records)
    inputs, targets, mask = next(loader.epoch())
    assert inputs.shape == targets.shape == mask.shape

    tok = ByteTokenizer()
    cleaned = [json.loads(line) for line in
               (corpus["root"] / "sft" / "cleaned" / "tool" / "records.jsonl"
                ).read_text().splitlines() if line.strip()]
    expected_by_ids = {}
    for row in cleaned:
        full_ids, spans = chat_template.response_spans(row["messages"], tok)
        want = [0] * (len(full_ids) - 1)
        for s, e in spans:
            for j in range(max(0, s - 1), min(e - 1, len(want))):
                want[j] = 1
        expected_by_ids[tuple(full_ids[:-1])] = want

    assert len(records) == len(cleaned)
    for i, rec in enumerate(records):
        n = len(rec["input_ids"])
        want = expected_by_ids[tuple(rec["input_ids"])]
        np.testing.assert_array_equal(mask[i, :n], np.asarray(want, dtype=np.float32))
        np.testing.assert_array_equal(mask[i, n:], 0.0)        # right-padding never trains

        trained = tok.decode([t for t, m in zip(rec["target_ids"], want) if m])
        untrained = tok.decode([t for t, m in zip(rec["target_ids"], want) if not m])
        # The assistant span, up to and including the stop token.
        assert trained.endswith(chat_template.CHAT_EOS)
        # Prompt, system tool declarations and tool results are all mask == 0.
        # (`target_ids` is `full_ids[1:]`, so the leading `<` is off the front.)
        assert "im_start|>system" in untrained and "<tools>" in untrained
        assert "<|im_start|>user" in untrained
        assert "<tool_response>" not in trained

    # The signal really does depend on the mask: with an all-zero mask the step is a no-op
    # (the existing `test_all_zero_mask_is_safe` invariant), so a mask that had silently
    # covered the prompt could not have produced the assertions above.
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    step = make_sft_train_step(model, optim.AdamW(learning_rate=1e-3), grad_clip=1.0)
    zeroed = step(model, [(inputs, targets, np.zeros_like(mask))], 1e-3)
    assert zeroed["loss"] == 0.0 and zeroed["grad_norm"] == 0.0
    real = step(model, [(inputs, targets, mask)], 1e-3)
    assert real["loss"] > 0.0 and real["grad_norm"] > 0.0


# --------------------------------------------------------------------------- #
# 4. tool edge cases reach a training batch
# --------------------------------------------------------------------------- #

def test_tool_abstention_and_multicall_reach_a_training_step(tmp_path):
    """An abstention row (no call) and a multi-call row both survive to a real update."""
    weather = {"name": "get_weather", "description": "Get current weather for a city",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                              "required": ["city"]}}
    clock = {"name": "get_time", "description": "Get the current time in a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                            "required": ["city"]}}
    rows = [
        build_tool_messages([weather, clock], "weather and time in Paris?",
                            [{"name": "get_weather", "arguments": {"city": "Paris"}},
                             {"name": "get_time", "arguments": {"city": "Paris"}}]),
        build_abstention_messages([weather], "translate hello into French",
                                  "I have no translation tool, but it is 'bonjour'."),
        build_tool_messages([weather], "weather in Oslo?",
                            [{"name": "get_weather", "arguments": {"city": "Oslo"}}]),
    ]
    m = build_tool_sft(rows, tmp_path, tokenizer=TOKENIZER, byte_fallback=True,
                       seq_len=SEQ_LEN, max_seq_len=SEQ_LEN)
    assert m["n_records"] == 3 and m["n_abstention"] == 1

    tok_dir = tmp_path / "sft" / "tokenized" / tokenized_dir_name(TOKENIZER, SEQ_LEN)
    on_disk = load_sft_records(tok_dir / "tool.jsonl")
    cfg = load_config(TOY_CFG)
    loader = SFTLoader(tok_dir, cfg.seq_len, 3, shuffle=False, drop_last=False,
                       vocab_size=cfg.vocab_size, records=on_disk)
    inputs, targets, mask = next(loader.epoch())
    tok = ByteTokenizer()
    decoded = [tok.decode([int(t) for t, mk in zip(targets[i], mask[i]) if mk])
               for i in range(3)]
    assert any(d.count("<tool_call>") == 2 for d in decoded)     # several calls
    assert any("<tool_call>" not in d for d in decoded)          # abstention
    assert all(d.endswith(chat_template.CHAT_EOS) for d in decoded)

    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    step = make_sft_train_step(model, optim.AdamW(learning_rate=1e-3), grad_clip=1.0)
    out = step(model, [(inputs, targets, mask)], 1e-3)
    assert np.isfinite(out["loss"]) and out["loss"] > 0.0
