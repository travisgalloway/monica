"""Scale corpus pipeline on datatrove (#80): the pod/cluster port of the local `src/data/` stages.

Runs the real datatrove `LocalPipelineExecutor` over a handful of synthetic docs (no network), so
the custom filter/scrub blocks and the MinHash dedup wiring are exercised end to end. datatrove is
absent in the main py3.14 env, so this whole module skips there and runs in the py3.11 `.venv-dt`:

    .venv-dt/bin/python -m pytest tests/test_datatrove_pipeline.py -q
"""

import glob
import gzip
import json

import pytest

pytest.importorskip("datatrove")

from src.data import datatrove_pipeline as dt

# Long, clean English paragraphs that clear the quality floor (>=200 chars, >=10 words).
GOOD = ("The selective state space model processes long sequences with linear memory, which makes "
        "very long context windows affordable on modest hardware during both training and serving. "
        "Because the recurrence is linear in sequence length, the model scales to documents that a "
        "quadratic attention transformer of the same size could never hold in memory at once.")
# Constructed at runtime so the full AWS-key pattern is never a literal in the repo (avoids
# tripping secret scanners) while still matching the scrubber's `\bAKIA[0-9A-Z]{16}\b`.
FAKE_AWS_KEY = "AKIA" + "1234567890ABCDEF"
SECRET = ("Configure the deployment with your credentials before the very first run of the worker. "
          f"For example, set the access key id {FAKE_AWS_KEY} in the environment, export the "
          "matching region and bucket name, and then start the worker process so it can connect to "
          "object storage and begin streaming the cleaned training shards from the remote prefix.")


def _reader(tmp_path, docs):
    from datatrove.pipeline.readers import JsonlReader
    d = tmp_path / "input"
    d.mkdir()
    with open(d / "data.jsonl", "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    return JsonlReader(str(d))


def _read_out(out_dir):
    docs = []
    for fp in sorted(glob.glob(f"{out_dir}/**/*.jsonl.gz", recursive=True)):
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            docs.extend(json.loads(line) for line in f)
    return docs


def test_clean_pipeline_filters_and_scrubs(tmp_path):
    docs = [
        {"text": GOOD, "id": "good", "metadata": {"source": "fineweb-edu", "license": "odc-by"}},
        {"text": "$$$ ### %%% @@@ &&& *** !!! ??? ~~~ ^^^ ||| \\\\\\ /// === +++ --- " * 6,
         "id": "junk", "metadata": {"source": "fineweb-edu", "license": "odc-by"}},
        {"text": "def add(a, b):\n    return a + b\n" * 4, "id": "gpl",
         "metadata": {"source": "the-stack", "lang": "python", "license": "gpl-3.0"}},
        {"text": "def add(a, b):\n    return a + b\n" * 4, "id": "mit",
         "metadata": {"source": "the-stack", "lang": "python", "license": "mit"}},
        {"text": SECRET, "id": "secret", "metadata": {"source": "fineweb-edu", "license": "odc-by"}},
    ]
    out = tmp_path / "out"
    pipe = dt.clean_pipeline(_reader(tmp_path, docs), out,
                             quality=True, license_filter=True, scrub=True)
    dt.make_executor(pipe, tmp_path / "logs", kind="local", tasks=1, workers=1).run()

    kept = {d["id"]: d for d in _read_out(f"{out}/cleaned")}
    # Quality drops the symbol-soup TEXT; the license gate drops the GPL CODE.
    assert "junk" not in kept and "gpl" not in kept
    # Clean text, permissive code, and the (scrubbed) secret doc are kept.
    assert set(kept) == {"good", "mit", "secret"}
    # The planted AWS key is redacted to the placeholder.
    assert FAKE_AWS_KEY not in kept["secret"]["text"]
    assert "[AWS_KEY]" in kept["secret"]["text"]
    assert kept["secret"]["metadata"].get("secrets_scrubbed", 0) >= 1


def test_minhash_dedup_removes_duplicate(tmp_path):
    # Two byte-identical docs + one distinct -> dedup keeps 2. Skips cleanly if the word tokenizer
    # needs assets unavailable offline.
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    docs = [
        {"text": GOOD, "id": "a", "metadata": {"source": "fineweb-edu"}},
        {"text": GOOD, "id": "b", "metadata": {"source": "fineweb-edu"}},
        {"text": SECRET, "id": "c", "metadata": {"source": "fineweb-edu"}},
    ]
    with open(cleaned / "data.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    try:
        dt.run_minhash_dedup(str(cleaned), str(tmp_path / "dd"), kind="local", tasks=1, workers=1)
    except LookupError as e:                       # e.g. nltk asset missing in an offline env
        pytest.skip(f"word-tokenizer asset unavailable offline: {e}")

    out = _read_out(f"{tmp_path}/dd/deduplicated")
    ids = {d["id"] for d in out}
    assert len(out) == 2 and "c" in ids           # one of the identical pair removed, distinct kept
