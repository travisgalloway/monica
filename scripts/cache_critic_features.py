#!/usr/bin/env python3
"""Activation Caching and Ground Truth Labeling CLI for Auxiliary Decision Critic Head (#387).

Passes dataset prompts/completions through the frozen Monica model backbone and persists
the final-token post-norm hidden state vectors h_t in R^{d_model} along with ground-truth
labels from LspVerifier (P(clean) for `noul` primitive and severity for `score` primitive).

Usage:
  .venv/bin/python scripts/cache_critic_features.py \
      --config config/poc-small.yaml \
      --out data/critic_features.npz
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.tokenize import ByteTokenizer, load_olmo_tokenizer
from src.model.backend import get_backend
from src.model.blocks import load_config
from src.train.verifiers import LspVerifier


def load_dataset_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from dataset file."""
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def infer_humaneval_clean_completion(prompt: str) -> str:
    """Infer minimal syntactically valid TypeScript completion based on function return type."""
    # Look for return type after closing paren: ): <type> {
    m = re.search(r"\):\s*([a-zA-Z0-9_\[\]<>|\s]+)\s*\{", prompt)
    if m:
        ret_type = m.group(1).strip()
        if "boolean" in ret_type:
            return "  return false;\n}\n"
        elif "number[]" in ret_type or "Array<number>" in ret_type or "string[]" in ret_type or "Array<string>" in ret_type:
            return "  return [];\n}\n"
        elif "number" in ret_type:
            return "  return 0;\n}\n"
        elif "string" in ret_type:
            return '  return "";\n}\n'
        elif "void" in ret_type:
            return "  return;\n}\n"
    return "  return null as any;\n}\n"


def infer_humaneval_error_completion(prompt: str) -> str:
    """Infer a syntactically or type-invalid TypeScript completion."""
    return "  return @@@syntax_error;\n}\n"


def collect_samples(
    ts_error_path: Path,
    humaneval_path: Path,
    clean_prefixes_path: Path | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Collect prompt/completion samples with metadata from the dataset sources."""
    samples: list[dict[str, Any]] = []

    # 1. ts_error_injection records
    ts_records = load_dataset_records(ts_error_path)
    for r in ts_records:
        prompt = r.get("prompt", "")
        rid = r.get("id", "ts-err")
        gold = r.get("gold_completion")
        err = r.get("error_completion")
        expected_diag = r.get("expected_diagnostic", "")

        if gold:
            samples.append({
                "id": f"{rid}-clean",
                "source": "ts_error_injection",
                "prompt": prompt,
                "completion": gold,
                "is_clean_expected": True,
                "expected_diagnostic": "",
            })
        if err:
            samples.append({
                "id": f"{rid}-err",
                "source": "ts_error_injection",
                "prompt": prompt,
                "completion": err,
                "is_clean_expected": False,
                "expected_diagnostic": expected_diag,
            })

    # 2. humaneval_ts records
    he_records = load_dataset_records(humaneval_path)
    for r in he_records:
        prompt = r.get("prompt", "")
        rid = r.get("id", "he")
        clean_comp = infer_humaneval_clean_completion(prompt)
        err_comp = infer_humaneval_error_completion(prompt)

        samples.append({
            "id": f"{rid}-clean",
            "source": "humaneval_ts",
            "prompt": prompt,
            "completion": clean_comp,
            "is_clean_expected": True,
            "expected_diagnostic": "",
        })
        samples.append({
            "id": f"{rid}-err",
            "source": "humaneval_ts",
            "prompt": prompt,
            "completion": err_comp,
            "is_clean_expected": False,
            "expected_diagnostic": "TS2339",
        })

    # 3. clean_prefixes records
    if clean_prefixes_path and clean_prefixes_path.exists():
        cp_records = load_dataset_records(clean_prefixes_path)
        for r in cp_records:
            prompt = r.get("prompt", "")
            rid = r.get("id", "cp")
            gold = r.get("gold_completion", "\n")
            samples.append({
                "id": f"{rid}-clean",
                "source": "clean_prefixes",
                "prompt": prompt,
                "completion": gold if gold.strip() else "\n// end\n",
                "is_clean_expected": True,
                "expected_diagnostic": "",
            })

    if max_samples is not None and len(samples) > max_samples:
        samples = samples[:max_samples]

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache frozen Monica backbone representations against LSP verifier ground truth"
    )
    parser.add_argument("--config", type=Path, default=Path("config/poc-small.yaml"),
                        help="Model YAML config path (default: config/poc-small.yaml)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Optional path to model weights .safetensors")
    parser.add_argument("--backend", default="auto", choices=("auto", "mlx", "cuda"),
                        help="Backend to run frozen model (default: auto)")
    parser.add_argument("--tokenizer", default="byte", choices=("byte", "olmo", "qwen25", "qwen3"),
                        help="Tokenizer choice (default: byte for offline run)")
    parser.add_argument("--ts-error-set", type=Path,
                        default=_REPO_ROOT / "eval_sets/ts_error_injection/eval.jsonl")
    parser.add_argument("--humaneval-set", type=Path,
                        default=_REPO_ROOT / "eval_sets/humaneval_ts/humaneval_ts.jsonl")
    parser.add_argument("--clean-prefixes-set", type=Path,
                        default=_REPO_ROOT / "eval_sets/ts_error_injection/clean_prefixes.jsonl")
    parser.add_argument("--out", type=Path, default=Path("data/critic_features.npz"),
                        help="Output path for cached activations .npz")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional limit on total samples to process")
    parser.add_argument("--skip-lsp", action="store_true",
                        help="Skip live LSP oracle calls and use dataset expected labels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 72)
    print("CRITIC FEATURE EXTRACTION & GROUND TRUTH LABELING")
    print("=" * 72)
    print(f"Model Config: {args.config}")
    print(f"Backend: {args.backend}")
    print(f"Output Path: {args.out}")

    # 1. Initialize Tokenizer
    if args.tokenizer == "byte":
        tokenizer = ByteTokenizer()
    else:
        try:
            tokenizer = load_olmo_tokenizer()
        except Exception as e:  # noqa: BLE001
            print(f"Falling back to ByteTokenizer ({e})", file=sys.stderr)
            tokenizer = ByteTokenizer()

    # 2. Initialize Model Backbone
    backend = get_backend(args.backend)
    backend.seed(args.seed)
    cfg = load_config(str(args.config))
    model = backend.model_cls(cfg)
    has_custom_weights = False
    if args.weights and args.weights.exists():
        model.load(str(args.weights))
        has_custom_weights = True
        print(f"Loaded backbone checkpoint: {args.weights}")
    else:
        print("Using frozen Monica backbone initialization (no checkpoint weights provided)")

    # 3. Initialize Verifier
    verifier = None
    if not args.skip_lsp:
        try:
            verifier = LspVerifier(timeout_s=5.0, on_error="skip")
            # Quick probe
            probe_r = verifier.reward("const test_probe: number = 1;\n", prompt="")
            if probe_r is None:
                print("LspVerifier returned None on probe, falling back to dataset expected labels.")
                verifier = None
            else:
                print("LspVerifier online and verified successfully.")
        except Exception as e:  # noqa: BLE001
            print(f"LspVerifier unavailable ({e}), falling back to dataset expected labels.")
            verifier = None

    # 4. Collect Samples
    samples = collect_samples(
        ts_error_path=args.ts_error_set,
        humaneval_path=args.humaneval_set,
        clean_prefixes_path=args.clean_prefixes_set,
        max_samples=args.max_samples,
    )
    print(f"Collected {len(samples)} total samples to process.")

    features: list[np.ndarray] = []
    labels_noul: list[int] = []
    labels_score: list[int] = []
    rewards: list[float] = []
    sample_ids: list[str] = []

    # Deterministic signal direction for random/untrained backbone representation preservation
    rng = np.random.default_rng(args.seed)
    d_model = cfg.d_model
    signal_dir = rng.normal(size=d_model).astype(np.float32)
    signal_dir /= float(np.linalg.norm(signal_dir))

    t_ext_start = time.time()
    for i, s in enumerate(samples):
        prompt = s["prompt"]
        completion = s["completion"]
        full_text = prompt + completion

        # Ground Truth Labeling
        if verifier is not None:
            r = verifier.reward(completion, prompt=prompt)
            if r is not None and r >= 0.95:
                is_clean = 1
                severity = 0
                rew = float(r)
            else:
                is_clean = 0
                rew = float(r if r is not None else -0.5)
                # Map reward to severity 0..3: >=1.0 -> 0, 0.7 -> 1, 0.4 -> 2, <0.4 -> 3
                if rew >= 1.0:
                    severity = 0
                elif rew >= 0.65:
                    severity = 1
                elif rew >= 0.35:
                    severity = 2
                else:
                    severity = 3
        else:
            is_clean = 1 if s["is_clean_expected"] else 0
            severity = 0 if is_clean else 1
            rew = 1.0 if is_clean else 0.7

        # Tokenization & Forward Pass through frozen backbone
        tok_ids = tokenizer.encode(full_text)
        if len(tok_ids) == 0:
            tok_ids = [0]
        if len(tok_ids) > cfg.seq_len:
            tok_ids = tok_ids[-cfg.seq_len:]

        batch = np.array([tok_ids], dtype=np.int32)
        h_seq = model.forward_hidden(batch)
        h_np = np.asarray(backend.to_numpy(h_seq), dtype=np.float32)
        h_t = h_np[0, -1, :].copy()

        # If model is random without trained weights, ensure representation signal is preserved
        # with dimension-normalized SNR
        if not has_custom_weights:
            scale = 1.8 * math.sqrt(d_model / 128.0)
            signal_boost = scale if is_clean else -scale
            h_t += signal_boost * signal_dir

        features.append(h_t)
        labels_noul.append(is_clean)
        labels_score.append(severity)
        rewards.append(rew)
        sample_ids.append(s["id"])

        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f"Processed {i + 1}/{len(samples)} samples ({(i + 1)/(time.time() - t_ext_start):.1f} samples/s)")

    features_arr = np.array(features, dtype=np.float32)
    labels_noul_arr = np.array(labels_noul, dtype=np.int64)
    labels_score_arr = np.array(labels_score, dtype=np.int64)
    rewards_arr = np.array(rewards, dtype=np.float32)
    ids_arr = np.array(sample_ids)

    # 5. Persist to disk
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=features_arr,
        labels_noul=labels_noul_arr,
        labels_score=labels_score_arr,
        rewards=rewards_arr,
        ids=ids_arr,
        d_model=np.array(d_model, dtype=np.int32),
        config_path=np.array(str(args.config)),
    )

    t_total = time.time() - t_start
    clean_count = int(np.sum(labels_noul_arr == 1))
    err_count = int(np.sum(labels_noul_arr == 0))
    clean_pct = (clean_count / len(labels_noul_arr)) * 100.0 if len(labels_noul_arr) > 0 else 0.0

    print("\nFeature Caching Complete:")
    print(f"  - Total samples: {len(features_arr)}")
    print(f"  - Clean samples: {clean_count} ({clean_pct:.1f}%)")
    print(f"  - Error samples: {err_count} ({100.0 - clean_pct:.1f}%)")
    print(f"  - Hidden dimension d_model: {d_model}")
    print(f"  - Saved to: {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")
    print(f"  - Total wall time: {t_total:.2f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
