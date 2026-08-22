"""Generate text from a trained Mamba-2/SSD checkpoint (MLX backend).

Wires the seam to the shared generation core: tokenizer encode -> SessionStore.step
-> sampler -> tokenizer decode, streaming tokens to stdout. Three modes:

  * completion (``--prompt "..."``): continue the raw prompt verbatim.
  * chat (``--chat``): a REPL that wraps each line in the SAME instruction template
    Dolly was formatted with at train time (``src.data.instruct_format``), so the
    model is prompted in the shape it learned. Generation stops at the next
    ``### Instruction:`` marker or EOS. Stateless by construction — a fresh session
    per turn, whole transcript re-rendered.
  * interactive (``--interactive``): a **stateful** continuation REPL. ONE session
    lives for the whole run; each line extends it and each completed exchange is
    committed to a ``RewindTree`` as a turn boundary, so ``/rewind`` restores the
    fixed-size recurrent state and branches history (#305).

mlx is imported inside ``main`` (local backend import, like ``scripts/smoke_test.py``)
so the file stays runnable to parse on non-Mac hosts — and so ``interactive_repl``
below is importable, and testable, without a backend.

    .venv/bin/python scripts/generate.py --config config/poc.yaml \\
        --weights runs/poc/weights.safetensors --prompt "The history of"
    .venv/bin/python scripts/generate.py --config config/poc.yaml \\
        --weights runs/poc/weights.safetensors --chat
    .venv/bin/python scripts/generate.py --config config/poc.yaml \\
        --weights runs/poc/weights.safetensors --interactive --temperature 0
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.data.instruct_format import INSTRUCTION_MARKER, render
from src.serve import sampling
from src.serve.generate import generate
from src.serve.sessions import SessionHistory, SessionStore

REPL_EPILOG = """\
interactive mode (--interactive) — one stateful session, rewindable:

  <text>          continue the session with <text> (the turn is committed as a
                  rewind boundary once its continuation is produced)
  /rewind [n]     rewind n retained turn boundaries (default 1) and branch there
  /rewind #<id>   rewind to an absolute node id, as printed by /tree
  /tree           list retained boundaries: id, parent, children, * = current
  /help           this list
  /quit           leave (Ctrl-D works too)

Retained boundaries are LRU-capped at --rewind-depth (default 32); the ceiling on
what they cost is depth x per_session_state_bytes(config), printed at startup.
--rewind-depth 0 disables rewind entirely (no tree is built).

Rewinds count RETAINED boundaries, not wall-clock turns: eviction reparents a node's
children onto its grandparent, so after the cap is hit one hop may skip a turn that
no longer exists.
"""


def _nonnegative_int(text: str) -> int:
    """argparse type for --rewind-depth: reject negatives loudly rather than clamp."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"must be >= 0 (0 disables rewind); got {value}")
    return value


def _parse_rewind_arg(arg: str) -> tuple[int, bool]:
    """`"2" -> (2, False)`, `"#7" -> (7, True)`. Raises ValueError on anything else."""
    body = arg[1:] if arg.startswith("#") else arg
    try:
        return int(body), arg.startswith("#")
    except ValueError:
        raise ValueError(
            "/rewind takes a turn count (e.g. `/rewind 2`) or an absolute node id "
            f"(e.g. `/rewind #3`); got {arg!r}") from None


def _path_to_root(nodes, node_id: int) -> list[int]:
    """Root-first ancestry of `node_id`, from a `SessionHistory.nodes()` listing."""
    parent_of = {nid: parent for nid, parent, _ in nodes}
    path: list[int] = []
    cursor: Optional[int] = node_id
    while cursor is not None and cursor in parent_of:
        path.append(cursor)
        cursor = parent_of[cursor]
    return list(reversed(path))


def interactive_repl(
    store,
    session_id: str,
    history: Optional[SessionHistory],
    *,
    run_turn: Callable[..., str],
    read_line: Callable[[], Optional[str]],
    emit: Callable[[str], None],
    rewind_enabled: bool = True,
) -> None:
    """Drive a stateful continuation session until `read_line()` returns None.

    Module-level and fully injected so a test can script stdin/stdout without a pty and
    without a backend: `run_turn(text, prefill=...) -> continuation` owns the tokenizer,
    sampler and `serve.generate.generate` call; `read_line()` yields the next input line
    (None at EOF); `emit(line)` writes one line of REPL chrome.

    The session is prefilled ONLY while it is genuinely fresh. After the first turn it has
    consumed tokens, and after a rewind its position is unknowable above the seam, so both
    cases fall back to `prefill=False` — `SessionStore.prefill` refuses them by design.

    The displayed transcript (`node id -> text`) lives here, not in `SessionHistory`: the
    tree holds opaque `State` blobs and text is a presentation concern. It is pruned to the
    tree's own retained ids on every commit so it stays bounded by `--rewind-depth` too —
    `RewindTree` evicts nodes but never tells this dict, so without pruning `transcript` would
    grow without bound over a long-lived session even though the state it describes does not.
    """
    transcript: dict[int, str] = {}

    def _prune_transcript() -> None:
        retained = set(history.retained_ids())
        for stale in [node_id for node_id in transcript if node_id not in retained]:
            del transcript[stale]

    root: Optional[int] = None                # this session's own turn-0 node id (#318)
    if rewind_enabled:
        if history is None:
            raise ValueError("rewind_enabled=True requires a SessionHistory")
        root = history.commit_turn()          # turn 0: the empty, freshly-created state
        transcript[root] = "(start)"
        emit(f"rewind: up to {history.max_depth} retained turn boundaries, at most "
             f"{history.budget_bytes()} bytes "
             f"({history.max_depth} x {history.per_node_bytes()} B/snapshot).")
    else:
        emit("rewind: DISABLED (--rewind-depth 0); no tree is built.")
    emit("interactive mode — type text to continue the session; /help for commands.")

    fresh = True
    turns_committed = 0                       # wall-clock turns, NOT retained depth (#318)
    while True:
        line = read_line()
        if line is None:
            break
        text = line.strip()
        if not text:
            continue

        if not text.startswith("/"):
            out = run_turn(text, prefill=fresh)
            fresh = False
            if rewind_enabled:
                node_id = history.commit_turn()
                turns_committed += 1
                transcript[node_id] = text + out
                _prune_transcript()
                emit(f"[committed #{node_id}; depth {history.depth()}, "
                     f"{len(history)}/{history.max_depth} retained]")
            continue

        parts = text.split()
        command, command_args = parts[0], parts[1:]

        if command in ("/quit", "/exit"):
            break

        if command == "/help":
            for help_line in REPL_EPILOG.splitlines():
                emit(help_line)
            continue

        if command == "/tree":
            if not rewind_enabled:
                emit("error: rewind is disabled (--rewind-depth 0); there is no tree.")
                continue
            emit(f"{len(history)}/{history.max_depth} retained boundaries, "
                 f"current depth {history.depth()}:")
            for node_id, parent, children in history.nodes():
                mark = "*" if node_id == history.current() else " "
                emit(f"{mark} #{node_id}  parent={parent}  children={children}  "
                     f"{transcript.get(node_id, '')!r}")
            continue

        if command == "/rewind":
            if not rewind_enabled:
                emit("error: rewind is disabled (--rewind-depth 0); rerun with "
                     "--rewind-depth N (N >= 1) to enable it.")
                continue
            if len(command_args) > 1:
                emit(f"error: /rewind takes at most one argument; got {command_args}")
                continue
            try:
                count, absolute = _parse_rewind_arg(
                    command_args[0] if command_args else "1")
                if absolute:
                    target = history.rewind_to(count)
                elif count == 1 and history.current() == root:
                    # DoD 6b: we are standing on THIS session's turn-0 node and the default
                    # `/rewind` (count 1) asked for exactly the one hop that doesn't exist.
                    # Restore the root (a real set_state, not a no-op) and say why there is
                    # nothing to rewind past.
                    #
                    # #318: the predicate keys on the root NODE ID, not on `history.depth()`.
                    # `depth()` counts *retained* ancestors, so `--rewind-depth 1` evicting
                    # turn 0 collapses it back to 1 while a real turn exists — which made this
                    # branch claim "session is empty" about a session with turns in it. The
                    # current node is never evicted and ids are never reused, so
                    # `current() == root` is exactly "on turn 0, and turn 0 is still retained".
                    # An evicted root therefore falls through to `rewind_turns` below and gets
                    # the precise hop-count error instead. So does a larger count (e.g.
                    # `/rewind 2` at the root), rather than being collapsed into this message
                    # regardless of how far past the root it asked to go.
                    target = history.rewind_to(root)
                    if turns_committed == 0:
                        emit("session is empty — nothing has been committed beyond the root "
                             "(turn 0); there is no earlier boundary to rewind past. "
                             "Restored the root state.")
                    else:
                        # Turns exist, we just rewound back onto turn 0. Not empty (#318).
                        emit("you are at the root (turn 0); there is no earlier boundary to "
                             "rewind past. Restored the root state.")
                else:
                    target = history.rewind_turns(count)
            except (ValueError, LookupError) as exc:
                emit(f"error: {exc}")          # no traceback; the session is untouched
                continue
            fresh = False                      # a restored session cannot be prefilled
            emit(f"rewound to #{target} (depth {history.depth()}); transcript:")
            for node_id in _path_to_root(history.nodes(), target):
                emit(f"  #{node_id}: {transcript.get(node_id, '')!r}")
            continue

        emit(f"error: unknown command {command!r} — /help lists the commands. It was "
             "NOT sent to the model.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, epilog=REPL_EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/poc-qwen.yaml"))
    ap.add_argument("--weights", type=Path, default=None,
                    help="portable .safetensors checkpoint; RANDOM INIT if omitted")
    ap.add_argument("--prompt", default=None, help="completion-mode prompt")
    ap.add_argument("--chat", action="store_true", help="instruction-template REPL")
    ap.add_argument("--interactive", action="store_true",
                    help="stateful continuation REPL with /rewind, /tree, /help (#305)")
    ap.add_argument("--rewind-depth", type=_nonnegative_int, default=32,
                    help="retained turn boundaries in --interactive mode (LRU-capped; "
                         "0 disables rewind, negative is an error)")
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
    ap.add_argument("--tokenizer", choices=("qwen3", "qwen25", "olmo"), default="qwen25",
                    help="tokenizer matching the config's vocab (default: qwen25, the "
                         "from-scratch poc-qwen reserve; use qwen3 for the distillation "
                         "student config/student-1b.yaml, olmo for config/poc.yaml)")
    ap.add_argument("--byte-fallback", action="store_true",
                    help="offline ByteTokenizer (toy config only; not OLMo/Qwen-compatible)")
    args = ap.parse_args()
    if args.chat and args.interactive:
        ap.error("--chat and --interactive are different REPLs; pick one")
    if not args.chat and not args.interactive and args.prompt is None:
        ap.error("provide --prompt for completion mode, --chat for the instruction "
                 "REPL, or --interactive for the stateful continuation REPL")

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
        load_qwen3_tokenizer,
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
    elif args.tokenizer == "qwen3":
        tok = load_qwen3_tokenizer()
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

    def encode(text: str) -> list[int]:
        # add_special_tokens=False for the same reason `run` uses it; ByteTokenizer.encode
        # takes no kwargs, hence the fallback.
        try:
            return tok.encode(text, add_special_tokens=False)
        except TypeError:
            return tok.encode(text)

    if args.interactive:
        sid = "repl"
        store.create(sid)
        history = (SessionHistory(store, sid, max_depth=args.rewind_depth)
                   if args.rewind_depth > 0 else None)

        def run_turn(text: str, *, prefill: bool = True) -> str:
            """Extend the live session with `text` and stream its continuation."""
            ids = encode(text)
            if not ids:
                return ""
            out_ids = generate(
                store, sid, ids, sampler=sampler, to_numpy=to_numpy,
                max_new_tokens=args.max_new_tokens, eos_id=eos_id,
                pass_context=True, prefill=prefill,
                on_token=lambda t: (sys.stdout.write(tok.decode([t])),
                                    sys.stdout.flush()),
            )
            print()
            return tok.decode(out_ids)

        def read_line() -> str | None:
            try:
                return input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                return None

        interactive_repl(store, sid, history, run_turn=run_turn, read_line=read_line,
                         emit=lambda s: print(s, file=sys.stderr),
                         rewind_enabled=args.rewind_depth > 0)
    elif args.chat:
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
