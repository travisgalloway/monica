# Custom correctness ruleset for opengrep (#199 Stage A)

`correctness.yaml` is a small, hand-authored opengrep/semgrep ruleset targeting
syntactically-recognizable TypeScript *logic* bugs — the class of defect a type checker
structurally cannot see (F1's finding: real HumanEval-TS bodies are 88.7% type-clean but only
50.3% correct). opengrep/semgrep is a **syntactic AST matcher with no type information**: it
catches recognizable bug *idioms*, not general off-by-one or algorithmic-misunderstanding errors.
A low hit-rate against real HumanEval-TS failures (mostly algorithmic, not idiomatic) is a
plausible, legitimate finding to report either way — not a reason to keep tuning the ruleset
after the fact.

## Pre-registration process (load-bearing methodology, per the plan)

Custom rules create circularity risk: rules written by inspecting which eval records fail make
"opengrep catches logic errors" self-fulfilling. Binding process followed for this ruleset:

1. Rules derived **only** from the general TypeScript bug taxonomy below — public prior
   knowledge, authored blind to this repo's eval outcomes.
2. `results/f1_base.jsonl` and every `results/*.jsonl` transcript were **not opened** during rule
   authoring.
3. Each rule ships a **positive + negative fixture** in `tests/test_opengrep.py`, drawn from the
   same taxonomy.
4. The ruleset is committed in its **own dedicated commit**, before any measurement, and is
   **frozen** at that commit for the eventual re-measure.
5. The `--limit 5` smoke run (plumbing verification only, done after this ruleset lands) must not
   feed back into rule selection.

## The 12 rules (+ 1 excluded)

| Rule | Confidence | Idiom |
|---|---|---|
| `loop-bound-off-by-one` | High | `for (let i = 0; i <= arr.length; i++)` |
| `index-at-length` | High | `arr[arr.length]` |
| `sort-without-comparator` | Medium (high match, medium value — fires on string arrays too) | `arr.sort()` |
| `indexof-truthy-check` | High | `arr.indexOf(x) > 0` |
| `parseint-no-radix` | High | `parseInt(x)` without a radix |
| `self-comparison` | High | `x == x` / `x != x` |
| `useless-ternary` | High | `c ? a : a` |
| `fill-shared-reference` | High | `Array(n).fill([])` / `.fill({})` |
| `strict-equality-nan` | High | `x === NaN` |
| `typeof-array` | High | `typeof x === "array"` |
| `assignment-in-condition` | Medium | `if (x = y) { ... }` |
| `for-in-over-array` | Medium | `for (const i in arr) { ...arr[i]... }` |
| ~~`map-without-return`~~ | — | **excluded**, see below |

Watch `over_repair_rate` for the medium-confidence rules in particular (`sort-without-comparator`,
`for-in-over-array`): a noisy rule becomes a rollback trigger that damages already-correct code.

## Toolchain (installed this Setup step)

- **opengrep `v1.25.0`**, binary `opengrep_osx_arm64`, installed to a directory on `PATH` (not
  vendored into the repo — same treatment as `node`/`npm` themselves).
- **cosign-verified** against the release's detached signature before install:

  ```bash
  cosign verify-blob opengrep_osx_arm64 \
    --certificate opengrep_osx_arm64.cert \
    --signature opengrep_osx_arm64.sig \
    --certificate-identity \
      "https://github.com/opengrep/opengrep/.github/workflows/rolling-release.yml@refs/heads/main" \
    --certificate-oidc-issuer "https://token.actions.githubusercontent.com"
  ```

  (`opengrep_osx_arm64.cert`/`.sig` are the base64 cert + signature assets published alongside
  the binary on the GitHub release — sigstore keyless signing, no static public key to pin.)

## `opengrep lsp` — verified empirically (undocumented, absent from `--help` output text but
listed as a subcommand)

- `initializationOptions` must nest scan settings under a `"scan"` key —
  `{"scan": {"configuration": [<rules dir>], "onlyGitDirty": false}}` — **not flat**. A flat
  `{"configuration": [...], "onlyGitDirty": false}` silently fails to configure anything (worse:
  an empty/missing `"scan"` object trips a real server-side deserialization bug —
  `Yojson.Safe.Util.Type_error("Can't get member 'pro_intrafile' of non-object type null")` —
  confirmed against the live `v1.25.0` binary).
- `onlyGitDirty: false` is mandatory: it defaults to `true`, and this project's LSP scratch dirs
  are `.gitignore`d, so git never reports them as "dirty" — with the default, the scanner sees
  zero targets, always.
- `didOpen` alone triggers a scan (a `textDocument/publishDiagnostics` notification for that uri
  follows); no `didSave` needed.
- A finding's `code` comes back as the **bare rule id** (e.g. `loop-bound-off-by-one`) when the
  configured rules directory sits inside a recognized git project (this repo's
  `eval_sets/opengrep_rules`) — confirmed against the real, final `correctness.yaml` in place.
  An early probe against a ruleset directory *outside* any git repo instead saw a
  `"<rules-dir-basename>.<rule-id>"` form (e.g. `probe_rules.probe-loop-off-by-one`); `opengrep.py`
  defensively strips a `f"{RULES_DIR.name}."` prefix if present, so either form maps to the same
  bare id. Both forms are safe against `is_incomplete`'s `^TS1\d{3}$` regex regardless.
- A genuinely unparseable/incomplete document (mid-generation prefix) produces a server-side
  parse error logged to stderr and an **empty** `publishDiagnostics` array — never a crash, never
  a hang. No extra incomplete-prefix handling is needed on the client side beyond what
  `diagnostics.py`'s frontier gating already does for the TS-LSP arm.
- The server sends server→client requests (`window/workDoneProgress/create`) during
  initialization/rescans; `jsonrpc.py`'s endpoint replies `{"result": null}` to any such request
  regardless of whether this particular exchange strictly requires it, so the loop can never
  block on one.

## Excluded rule

`map-without-return` (a callback passed to `.map()` that never executes a `return`) is **not
reliably expressible** as a syntactic AST pattern — semgrep/opengrep has no data-flow reasoning
about which paths through a callback body execute a `return`, so any pattern attempt either
matches almost every `.map()` call (uselessly noisy) or almost none (uselessly narrow). Excluded
from the 12-rule set; recorded here rather than silently dropped.
