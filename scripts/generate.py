"""Generate text from a trained Mamba-2/SSD checkpoint (MLX backend).

Wires the seam to the shared generation core: tokenizer encode -> SessionStore.step
-> sampler -> tokenizer decode, streaming tokens to stdout. Two modes:

  * completion (``--prompt "..."``): continue the raw prompt verbatim.
  * chat (``--chat``): a REPL that wraps each line in the SAME instruction template
    Dolly was formatted with at train time (``src.data.instruct_format``), so the
    model is prompted in the shape it learned. Generation stops at the next
    ``### Instruction:`` marker or EOS.

mlx is imported inside ``main`` (local backend import, like ``scripts/smoke_test.py``)
so the file stays runnable to parse on non-Mac hosts.

    .venv/bin/python scripts/generate.py --config config/poc.yaml \\
        --weights runs/poc/weights.safetensors --prompt "The history of"
    .venv/bin/python scripts/generate.py --config config/poc.yaml \\
        --weights runs/poc/weights.safetensors --chat
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np

from src.data.instruct_format import INSTRUCTION_MARKER, render
from src.serve import sampling
from src.serve.generate import generate
from src.serve.sessions import SessionStore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/poc.yaml"))
    ap.add_argument("--weights", type=Path, default=None,
                    help="portable .safetensors checkpoint; RANDOM INIT if omitted")
    ap.add_argument("--prompt", default=None, help="completion-mode prompt")
    ap.add_argument("--chat", action="store_true", help="instruction-template REPL")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="CTRL-style penalty on already-seen tokens (1.0 = off; "
                         "~1.2-1.3 curbs the 'United States, the United States…' loops)")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=None,
                    help="hard-ban repeating any n-gram of this size (e.g. 3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", choices=("qwen25", "olmo"), default="qwen25",
                    help="tokenizer matching the config's vocab (default: qwen25; "
                         "use olmo for the OLMo-vocab config/poc.yaml)")
    ap.add_argument("--byte-fallback", action="store_true",
                    help="offline ByteTokenizer (toy config only; not OLMo/Qwen-compatible)")
    args = ap.parse_args()
    if not args.chat and args.prompt is None:
        ap.error("provide --prompt for completion mode, or --chat for the REPL")

    try:
        import mlx.core as mx  # noqa: F401  (seeded below; import proves availability)
    except ModuleNotFoundError as e:
        if e.name != "mlx":
            raise
        raise SystemExit(
            "mlx not found — run with the project venv on Apple Silicon:\n"
            "    .venv/bin/python scripts/generate.py ...")
    from src.data.tokenize import (
        ByteTokenizer,
        load_olmo_tokenizer,
        load_qwen25_tokenizer,
    )
    from src.model.blocks import load_config
    from src.model.mlx_backend import MLXMambaModel

    cfg = load_config(str(args.config))
    mx.random.seed(args.seed)
    model = MLXMambaModel(cfg)
    if args.weights:
        model.load(str(args.weights))
        print(f"loaded weights: {args.weights}", file=sys.stderr)
    else:
        print("RANDOM INIT — no --weights; output will be gibberish.", file=sys.stderr)

    if args.byte_fallback:
        tok = ByteTokenizer()
    elif args.tokenizer == "qwen25":
        tok = load_qwen25_tokenizer()
    else:
        tok = load_olmo_tokenizer()
    if tok.vocab_size > cfg.vocab_size:
        raise SystemExit(
            f"tokenizer vocab {tok.vocab_size} exceeds model vocab {cfg.vocab_size} "
            f"({args.config}) — use --byte-fallback only with toy-scale configs.")

    eos_id = getattr(tok, "eos_token_id", None)
    rng = np.random.default_rng(args.seed)
    sampler = partial(sampling.sample, temperature=args.temperature,
                      top_k=args.top_k, top_p=args.top_p, rng=rng,
                      repetition_penalty=args.repetition_penalty,
                      no_repeat_ngram_size=args.no_repeat_ngram_size)
    store = SessionStore(model, max_concurrent=1)
    to_numpy = lambda a: np.array(a)

    def run(text: str, *, stop_marker: str | None) -> str:
        """Encode `text`, emit the continuation to stdout, on one fresh session.

        Completion mode (no stop marker) streams token-by-token. Chat mode buffers and
        prints once: the stop marker is only detectable *after* its tokens are produced,
        so streaming would leak the marker to stdout before it could be truncated.
        Returns the produced text (truncated at the stop marker in chat mode) so the
        REPL can append it to the running conversation history.
        """
        # add_special_tokens=False: the HF tokenizer otherwise appends EOS to the
        # prompt, which makes the model immediately predict end-of-text and emit
        # nothing. ByteTokenizer.encode takes no kwargs, hence the fallback.
        try:
            ids = tok.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tok.encode(text)
        if not ids:
            return ""
        sid = "cli"
        store.create(sid)
        produced = ""
        try:
            if stop_marker is None:
                generate(
                    store, sid, ids, sampler=sampler, to_numpy=to_numpy,
                    max_new_tokens=args.max_new_tokens, eos_id=eos_id,
                    pass_context=True,
                    on_token=lambda t: (sys.stdout.write(tok.decode([t])),
                                        sys.stdout.flush()),
                )
            else:
                out_ids = generate(
                    store, sid, ids, sampler=sampler, to_numpy=to_numpy,
                    max_new_tokens=args.max_new_tokens, eos_id=eos_id,
                    pass_context=True,
                    stop_fn=lambda gen: stop_marker in tok.decode(gen),
                )
                out = tok.decode(out_ids)
                cut = out.find(stop_marker)
                produced = out if cut == -1 else out[:cut]
                sys.stdout.write(produced)
        finally:
            store.remove(sid)
        print()
        return produced

    if args.chat:
        print("chat mode — type a message (Ctrl-D / Ctrl-C to exit).", file=sys.stderr)
        messages: list[dict] = []
        while True:
            try:
                line = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                break
            if not line.strip():
                continue
            messages.append({"role": "user", "content": line})
            # Prompt up to (and including) a trailing empty "### Response:" marker; the
            # model fills in the answer. A first turn renders to format_prompt(line).
            prompt = render(messages + [{"role": "assistant", "content": ""}])
            reply = run(prompt, stop_marker=INSTRUCTION_MARKER)
            messages.append({"role": "assistant", "content": reply.strip()})
    else:
        sys.stdout.write(args.prompt)
        run(args.prompt, stop_marker=None)


if __name__ == "__main__":
    main()
