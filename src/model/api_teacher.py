"""Top-k-logit-only conversion teacher over an OpenAI-compatible endpoint (LM Studio, llama.cpp
`--server`, vLLM, ...). PORTABLE — pure stdlib + numpy + the HF tokenizer, no `mlx`/`torch` — so
it stays above the seam and in `tests/test_import_guard.py`'s portable set.

WHY THIS IS PARTIAL (read before using). A distillation conversion teacher is a frozen, white-box
model: the student init (#99) reads its Q/K/V/O projections and embeddings, and the matching
stages (#100) read per-layer hidden states (`hidden-align`) and attention matrices
(`mixing-match`). A chat/completions HTTP endpoint exposes NONE of that — only per-token output
logprobs. So this teacher implements ONLY `topk_logits`, which is enough to drive the **teacher
top-k precompute (#94) -> the `logit-distill` stage (#100)** and nothing else. For real local
validation prefer the full-fidelity MLX teacher (`precompute_teacher.py --backend mlx --pretrained
Qwen/Qwen3-4B-Thinking-2507`, which runs the actual weights on Apple Silicon). Use this endpoint
path only as a convenience when a model is already loaded in LM Studio.

KNOWN APPROXIMATIONS (all documented, none silent):
  * **logprobs, not logits.** Servers return log-probabilities (log-softmax over the full vocab).
    The distill `_kl_topk` re-softmaxes the cached top-k at temperature T; at T==1 a constant
    offset cancels so top-k logprobs == top-k logits, but at T!=1 the scaling is only approximate.
  * **capped k.** The OpenAI `logprobs`/`top_logprobs` count is server-capped (often <=10-20), so
    the effective cached k can be smaller than requested.
  * **string->id remap.** Endpoints return top tokens as TEXT; we map them back to vocab ids with
    the Qwen3 tokenizer best-effort (`convert_tokens_to_ids`, then a single-token `encode`
    fallback). Unmappable entries are dropped (logged once).
  * **re-tokenization alignment.** We must send the prompt as TEXT (ids are detokenized first),
    and the server re-tokenizes; if its segmentation differs from the packed ids the per-position
    alignment drifts. We align by position up to `seq_len`, padding/truncating, and warn once.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .teacher import AttnProjections, ConversionTeacher

_NEG_INF = float(np.finfo(np.float32).min)


@dataclass
class _ApiTeacherConfig:
    """Minimal config the precompute driver reads (`vocab_size`, `effective_vocab_size`). The
    endpoint teacher emits ids already clipped to the student tokenizer vocab, so the two match."""
    vocab_size: int
    n_layers: int = 0

    @property
    def effective_vocab_size(self) -> int:
        return self.vocab_size


def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    """POST `payload` as JSON to `url`, return the parsed JSON response (stdlib only)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 (local trusted URL)
        return json.loads(resp.read().decode("utf-8"))


class ApiTopkTeacher(ConversionTeacher):
    """Frozen top-k-logit-only teacher backed by an OpenAI-compatible completions endpoint.

    Only `topk_logits` is supported; the white-box methods (`forward`, `attention_projection`,
    `embedding_matrix`, `lm_head_matrix`) raise `NotImplementedError` pointing at `--pretrained`.
    `_post`/`_tokenizer` are injectable seams so the unit tests run offline (no server, no HF).
    """

    def __init__(self, *, base_url: str, vocab_size: int, tokenizer: str = "qwen3",
                 model: Optional[str] = None, timeout: float = 120.0,
                 _post: Optional[Callable[[str, dict, float], dict]] = None,
                 _tokenizer: Any = None):
        if tokenizer != "qwen3" and _tokenizer is None:
            raise ValueError(f"ApiTopkTeacher supports the qwen3 tokenizer (got {tokenizer!r}); the "
                             "endpoint must serve a Qwen3-tokenizer model for id alignment")
        self.config = _ApiTeacherConfig(vocab_size=int(vocab_size))
        self._url = base_url.rstrip("/") + "/completions"
        self._model = model
        self._timeout = timeout
        self._post = _post or _http_post_json
        if _tokenizer is not None:
            self._tok = _tokenizer
        else:
            from src.data.tokenize import load_qwen3_tokenizer   # lazy: HF, not in offline tests
            self._tok = load_qwen3_tokenizer()
        self._warned_align = False

    # --- ConversionTeacher: the one supported method -------------------------
    def topk_logits(self, token_batch, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Per-token top-`k` (values, indices), each `(B, L, k)`, from the endpoint's logprobs.

        Values are descending log-probs (see module note); padding for missing/unmappable entries
        is `-inf` value / `0` index. Rows are processed one prompt at a time (echo + logprobs)."""
        rows = np.asarray(token_batch)
        if rows.ndim == 1:
            rows = rows[None, :]
        B, L = rows.shape
        vals = np.full((B, L, k), _NEG_INF, dtype=np.float32)
        idx = np.zeros((B, L, k), dtype=np.int64)
        for b in range(B):
            v_row, i_row = self._row_topk(rows[b].tolist(), L, k)
            vals[b], idx[b] = v_row, i_row
        return vals, idx

    def _row_topk(self, ids: List[int], L: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        prompt = self._tok.decode(ids)
        payload = {"prompt": prompt, "max_tokens": 0, "echo": True, "logprobs": k,
                   "temperature": 0.0}
        if self._model is not None:
            payload["model"] = self._model
        resp = self._post(self._url, payload, self._timeout)
        top_list = self._extract_top_logprobs(resp)
        if len(top_list) != L and not self._warned_align:
            print(f"ApiTopkTeacher: server re-tokenized prompt to {len(top_list)} positions != "
                  f"seq_len {L}; aligning by position (truncate/pad). Top-k alignment is "
                  f"APPROXIMATE — prefer --pretrained for exactness.", file=sys.stderr)
            self._warned_align = True
        vals = np.full((L, k), _NEG_INF, dtype=np.float32)
        idx = np.zeros((L, k), dtype=np.int64)
        for pos in range(min(L, len(top_list))):
            pairs = self._map_tokens(top_list[pos])          # [(id, logprob), ...] descending
            for j, (tid, lp) in enumerate(pairs[:k]):
                vals[pos, j], idx[pos, j] = lp, tid
        return vals, idx

    @staticmethod
    def _extract_top_logprobs(resp: dict) -> List[dict]:
        """Pull the per-position `{token_str: logprob}` dicts from an OpenAI completions response."""
        choices = resp.get("choices") or [{}]
        lp = choices[0].get("logprobs") or {}
        top = lp.get("top_logprobs")
        return list(top) if top else []

    def _map_tokens(self, top: dict) -> List[Tuple[int, float]]:
        """Map a `{token_str: logprob}` dict to `[(vocab_id, logprob)]`, descending, dropping
        unmappable tokens and ids outside the student tokenizer vocab."""
        out: List[Tuple[int, float]] = []
        for tok_str, lp in top.items():
            tid = self._token_to_id(tok_str)
            if tid is not None and 0 <= tid < self.config.vocab_size:
                out.append((tid, float(lp)))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def _token_to_id(self, tok_str: str) -> Optional[int]:
        tid = self._tok.convert_tokens_to_ids(tok_str)
        unk = getattr(self._tok, "unk_token_id", None)
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            return tid
        enc = self._tok.encode(tok_str, add_special_tokens=False)
        return enc[0] if len(enc) == 1 else None

    def to_numpy(self, array) -> np.ndarray:
        return np.asarray(array)

    # --- ConversionTeacher: the white-box methods this path cannot provide ---
    def forward(self, token_batch, *, return_hidden: bool = False):
        raise NotImplementedError(self._unsupported("forward / hidden states"))

    def attention_projection(self, layer: int) -> AttnProjections:
        raise NotImplementedError(self._unsupported("attention projections (#99 init)"))

    def embedding_matrix(self):
        raise NotImplementedError(self._unsupported("embedding matrix (#99 init)"))

    def lm_head_matrix(self):
        raise NotImplementedError(self._unsupported("lm-head matrix (#99 init)"))

    @staticmethod
    def _unsupported(what: str) -> str:
        return (f"ApiTopkTeacher exposes top-k logits only — {what} needs the model's weights, "
                "which an OpenAI-compatible endpoint does not expose. Use a white-box teacher: "
                "precompute_teacher.py --backend mlx --pretrained <Qwen3 id>.")
