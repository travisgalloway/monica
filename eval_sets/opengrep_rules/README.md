# Custom correctness ruleset for opengrep (#199 Stage A)

**Status: scaffold.** This directory will hold `correctness.yaml`, a small, hand-authored
opengrep/semgrep ruleset targeting syntactically-recognizable TypeScript *logic* bugs — the
class of defect a type checker structurally cannot see (F1's finding: real HumanEval-TS bodies
are 88.7% type-clean but only 50.3% correct). The ruleset and its fixtures
(`tests/test_opengrep.py`) land in their own dedicated commit, authored **blind** to this repo's
eval transcripts (`results/*.jsonl`) — see the pre-registration process below once the ruleset
commit lands.

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
- A finding's `code` comes back as `"<rules-dir-basename>.<rule-id>"` (e.g.
  `opengrep_rules.loop-bound-off-by-one`), not the bare rule id — `opengrep.py` strips the
  known `f"{RULES_DIR.name}."` prefix so the recorded `Diagnostic.code` is the bare rule id.
  Either form is safe against `is_incomplete`'s `^TS1\d{3}$` regex.
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
