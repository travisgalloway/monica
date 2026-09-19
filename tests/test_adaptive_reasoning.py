"""Tests for the adaptive dual-mode reasoning gate (#362).

Validates:
1. Mode conventions in `src/data/chat_template.py` (direct vs reasoning).
2. Suppression of `<think>` tokens during inline FIM completion.
3. Structured multi-step reasoning traces permitted during instruction tasks.
4. Bounding of reasoning traces by configured `max_reasoning_tokens` limits.
5. Deterministic mode switching across repeated runs.
6. Direct cursor insertion for FIM in `scripts/generate.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.data.chat_template import (
    FIM_PREFIX,
    MODE_DIRECT,
    MODE_REASONING,
    SYSTEM_PROMPT_DIRECT,
    SYSTEM_PROMPT_REASONING,
    THINK_END,
    THINK_START,
    VALID_MODES,
    format_mode_messages,
    get_mode_for_prompt,
    is_fim_prompt,
    render,
)
from src.serve.generate import AdaptiveReasoningGate, generate
from src.serve.sampling import sample
from src.serve.sessions import SessionStore


# --------------------------------------------------------------------------------------- #
# Deterministic mock model for mode testing
# --------------------------------------------------------------------------------------- #

class ReasoningMockModel:
    """Mock model where token 10 is <think>, token 11 is </think>, and other tokens count up.

    If current token is 0 or user text, it eagerly predicts <think> (token 10) with highest logit.
    Inside <think> (after token 10), it predicts reasoning tokens (5, 6, 7...) until </think> (token 11).
    After </think>, it predicts answer tokens (20, 21, 22...).
    """

    def __init__(self, vocab_size: int = 32):
        self.config = SimpleNamespace(
            n_layers=2, d_conv=4, d_inner=128, n_heads=8, head_dim=16, d_state=16,
            precision="fp32", vocab_size=vocab_size,
        )
        self.vocab_size = vocab_size

    def init_state(self, batch_size: int):
        return np.zeros((batch_size,), dtype=np.int64)

    def step(self, token, state):
        tok = int(np.asarray(token).reshape(-1)[0])
        logits = np.zeros((1, self.vocab_size), dtype=np.float32)

        # Natural preference: try to output <think> (token 10) first, or continue reasoning
        if tok == 10:  # just emitted <think>
            logits[0, 5] = 10.0  # emit reasoning token 5
        elif tok in (5, 6, 7, 8):  # in reasoning
            next_reas = tok + 1 if tok < 8 else 11  # transition to </think> (11)
            logits[0, next_reas] = 10.0
        elif tok == 11:  # just emitted </think>
            logits[0, 20] = 10.0  # answer token
        elif 20 <= tok < 30:
            logits[0, min(self.vocab_size - 1, tok + 1)] = 10.0
        else:
            # Default first token: wants to emit <think> (token 10) with top score,
            # but fallback token 20 has second highest score.
            logits[0, 10] = 10.0
            logits[0, 20] = 5.0

        return logits, state + tok

    def prefill(self, token_batch, seg_ids=None, *, last_only=False):
        token_batch = np.asarray(token_batch)
        state = self.init_state(token_batch.shape[0])
        rows = []
        for t in range(token_batch.shape[1]):
            logits, state = self.step(token_batch[:, t], state)
            rows.append(logits)
        return (rows[-1] if last_only else np.stack(rows, axis=1)), state

    def clone_state(self, state):
        return np.array(state, copy=True)


def _make_store():
    store = SessionStore(ReasoningMockModel())
    store.create("test_session")
    return store


# --------------------------------------------------------------------------------------- #
# 1. Mode Conventions in chat_template.py
# --------------------------------------------------------------------------------------- #

def test_chat_template_mode_constants():
    assert MODE_DIRECT == "direct"
    assert MODE_REASONING == "reasoning"
    assert VALID_MODES == ("direct", "reasoning")
    assert THINK_START == "<think>"
    assert THINK_END == "</think>"
    assert FIM_PREFIX == "<|fim_prefix|>"


def test_is_fim_prompt_detection():
    # String prompt
    assert is_fim_prompt("<|fim_prefix|>def foo():\n<|fim_suffix|>\n<|fim_middle|>")
    assert not is_fim_prompt("def foo():\n    return 42")

    # Sentinel token id (1)
    assert is_fim_prompt([1, 42, 43])
    assert not is_fim_prompt([2, 42, 43])

    # Byte-encoded FIM prefix
    fim_bytes = list(FIM_PREFIX.encode("utf-8"))
    assert is_fim_prompt(fim_bytes + [32, 33])


def test_get_mode_for_prompt():
    assert get_mode_for_prompt("<|fim_prefix|>code<|fim_suffix|>") == MODE_DIRECT
    assert get_mode_for_prompt("<|im_start|>user\nrefactor this<|im_end|>") == MODE_REASONING
    assert get_mode_for_prompt("plain text", default_mode=MODE_DIRECT) == MODE_DIRECT


def test_format_mode_messages_helpers():
    msgs = [{"role": "user", "content": "hello"}]
    direct_msgs = format_mode_messages(msgs, mode=MODE_DIRECT)
    assert direct_msgs[0]["role"] == "system"
    assert direct_msgs[0]["content"] == SYSTEM_PROMPT_DIRECT

    reas_msgs = format_mode_messages(msgs, mode=MODE_REASONING)
    assert reas_msgs[0]["role"] == "system"
    assert reas_msgs[0]["content"] == SYSTEM_PROMPT_REASONING

    with pytest.raises(ValueError, match="unknown mode"):
        format_mode_messages(msgs, mode="invalid_mode")


def test_render_with_mode_conventions():
    msgs = [{"role": "user", "content": "implement quicksort"}]

    direct_out = render(msgs, mode=MODE_DIRECT, add_generation_prompt=True)
    assert SYSTEM_PROMPT_DIRECT in direct_out
    assert direct_out.endswith("<|im_start|>assistant\n")

    reas_out = render(msgs, mode=MODE_REASONING, add_generation_prompt=True)
    assert SYSTEM_PROMPT_REASONING in reas_out
    assert direct_out != reas_out

    # If system message is already present, custom instructions are preserved
    custom_msgs = [
        {"role": "system", "content": "custom prompt"},
        {"role": "user", "content": "hi"},
    ]
    rendered_custom = render(custom_msgs, mode=MODE_REASONING)
    assert "custom prompt" in rendered_custom
    assert SYSTEM_PROMPT_REASONING not in rendered_custom


# --------------------------------------------------------------------------------------- #
# 2. AdaptiveReasoningGate class unit tests
# --------------------------------------------------------------------------------------- #

def test_adaptive_reasoning_gate_direct_mode():
    gate = AdaptiveReasoningGate(
        prompt_ids=[1, 2, 3],
        fim_prefix_ids=1,
        think_token_ids=10,
        think_close_ids=11,
    )
    assert gate.is_fim is True
    assert gate.mode == "direct"

    logits = np.array([1.0] * 20, dtype=np.float32)
    filtered = gate.filter_logits(logits, generated=[])
    assert filtered[10] == -np.inf
    assert filtered[0] == 1.0


def test_adaptive_reasoning_gate_reasoning_mode_and_call():
    gate = AdaptiveReasoningGate(
        prompt_ids=[2, 3],
        fim_prefix_ids=1,
        think_token_ids=10,
        think_close_ids=11,
        max_reasoning_tokens=2,
    )
    assert gate.is_fim is False
    assert gate.mode == "reasoning"

    logits = np.array([1.0] * 20, dtype=np.float32)
    # Allows think token initially
    filtered = gate.filter_logits(logits, generated=[])
    assert filtered[10] == 1.0

    # Emit think token
    gate.record_token(10)
    assert gate.in_reasoning is True

    # Emit 2 reasoning tokens
    gate.record_token(5)
    gate.record_token(6)
    assert gate.reasoning_tokens_count == 2

    # Now limit reached: forces close (11)
    filtered_limit = gate.filter_logits(logits, generated=[10, 5, 6])
    assert filtered_limit[11] == 100.0
    assert filtered_limit[10] == -np.inf

    # Gate can also be called as a hook with previous_tokens
    hook_filtered = gate(logits, previous_tokens=[2, 3, 10, 5, 6])
    assert hook_filtered[11] == 100.0


# --------------------------------------------------------------------------------------- #
# 3. Inference Filtering: FIM suppresses <think> tokens
# --------------------------------------------------------------------------------------- #

def test_fim_suppresses_think_tokens():
    """When a FIM sentinel is present, the gate suppresses <think> tokens completely."""
    store = _make_store()
    greedy = partial(sample, temperature=0.0)

    # Prompt with FIM prefix sentinel (1)
    fim_prompt = [1, 2, 3]
    think_token_id = 10
    think_close_id = 11

    out = generate(
        store,
        "test_session",
        fim_prompt,
        sampler=greedy,
        max_new_tokens=4,
        think_token_id=think_token_id,
        think_close_id=think_close_id,
        fim_prefix_id=1,
    )

    # Model wanted to emit token 10 (<think>), but in FIM mode it was suppressed
    # and fallback token 20 was chosen instead.
    assert think_token_id not in out
    assert out[0] == 20
    assert 10 not in out  # Zero <think> tokens produced


def test_instruction_permits_reasoning_tokens():
    """When FIM sentinel is absent (instruction mode), <think> tokens are permitted."""
    store = _make_store()
    greedy = partial(sample, temperature=0.0)

    # Prompt without FIM sentinel
    instruct_prompt = [99, 100]
    think_token_id = 10
    think_close_id = 11

    out = generate(
        store,
        "test_session",
        instruct_prompt,
        sampler=greedy,
        max_new_tokens=6,
        think_token_id=think_token_id,
        think_close_id=think_close_id,
        fim_prefix_id=1,
    )

    # In instruction mode, <think> is permitted
    assert out[0] == 10  # <think> emitted
    assert 5 in out  # reasoning token emitted


# --------------------------------------------------------------------------------------- #
# 4. Reasoning trace bounded by configured token limits
# --------------------------------------------------------------------------------------- #

def test_reasoning_token_limit_bounds_trace():
    """Reasoning token limit bounds trace and cleanly closes </think> when limit reached."""
    store = _make_store()
    greedy = partial(sample, temperature=0.0)

    instruct_prompt = [99, 100]
    think_token_id = 10
    think_close_id = 11

    # Cap reasoning tokens inside <think>...</think> to at most 2 tokens
    out = generate(
        store,
        "test_session",
        instruct_prompt,
        sampler=greedy,
        max_new_tokens=6,
        think_token_id=think_token_id,
        think_close_id=think_close_id,
        max_reasoning_tokens=2,
        fim_prefix_id=1,
    )

    # Output should start with <think> (10), followed by at most 2 reasoning tokens,
    # then forced </think> (11), then answer tokens.
    assert out[0] == 10
    # Find position of </think> (11)
    close_pos = out.index(11)
    # Number of reasoning tokens between <think> and </think> is close_pos - 1
    reasoning_token_count = close_pos - 1
    assert reasoning_token_count <= 2
    # After </think>, <think> must not appear again
    assert 10 not in out[close_pos + 1 :]


# --------------------------------------------------------------------------------------- #
# 5. Deterministic Mode Switching
# --------------------------------------------------------------------------------------- #

def test_deterministic_mode_switching_identical_runs():
    """Two generation runs under the same mode produce identical token streams."""
    store = _make_store()
    greedy = partial(sample, temperature=0.0)

    # Direct FIM run
    fim_prompt = [1, 50]
    out_fim_1 = generate(store, "test_session", fim_prompt, sampler=greedy, max_new_tokens=5,
                         think_token_id=10, think_close_id=11, fim_prefix_id=1)
    store.create("test_session_2")
    out_fim_2 = generate(store, "test_session_2", fim_prompt, sampler=greedy, max_new_tokens=5,
                         think_token_id=10, think_close_id=11, fim_prefix_id=1)
    assert out_fim_1 == out_fim_2
    assert 10 not in out_fim_1

    # Instruction run
    inst_prompt = [2, 50]
    store.create("test_session_3")
    out_inst_1 = generate(store, "test_session_3", inst_prompt, sampler=greedy, max_new_tokens=5,
                          think_token_id=10, think_close_id=11, fim_prefix_id=1)
    store.create("test_session_4")
    out_inst_2 = generate(store, "test_session_4", inst_prompt, sampler=greedy, max_new_tokens=5,
                          think_token_id=10, think_close_id=11, fim_prefix_id=1)
    assert out_inst_1 == out_inst_2
    assert out_inst_1[0] == 10  # <think> emitted in instruction mode


# --------------------------------------------------------------------------------------- #
# 6. scripts/generate.py FIM direct cursor insertion
# --------------------------------------------------------------------------------------- #

def test_scripts_generate_cli_fim_direct_insertion():
    """FIM completions in scripts/generate.py insert directly without echoing prompt framing."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/generate.py"),
            "--byte-fallback",
            "--config", "config/toy-hybrid.yaml",
            "--fim-prefix", "def add(a, b):\n",
            "--fim-suffix", "\n    return res",
            "--max-new-tokens", "10",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    # The output should NOT start with <|fim_prefix|> echoing to stdout
    assert not res.stdout.startswith("<|fim_prefix|>")
    # Output must contain zero <think> tokens
    assert "<think>" not in res.stdout


# --------------------------------------------------------------------------------------- #
# 7. Evaluation suite: accuracy versus token consumption across both modes
# --------------------------------------------------------------------------------------- #

def test_eval_code_suite_adaptive_reasoning_suite():
    """scripts/eval_code_suite.py measures direct vs reasoning accuracy and tokens."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/eval_code_suite.py"),
            "--stub-model",
            "--byte-tokenizer",
            "--suites", "adaptive_reasoning",
            "--limit", "2",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "adaptive_reasoning: direct completion vs reasoning trace" in res.stdout
    assert "direct" in res.stdout
    assert "reasoning" in res.stdout
