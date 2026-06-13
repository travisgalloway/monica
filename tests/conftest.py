"""Shared test fixtures.

`toy_train_bin` builds a hermetic, learnable byte-fallback corpus so train-step tests
never read the shared, gitignored `data/split/train.bin` — that path is also written by
the real POC pipeline (`src.data.split`), so after a real data-prep run it holds
OLMo-tokenized ids (vocab 50280). Feeding those into the toy model (vocab 256) caused
out-of-bounds gathers → non-finite grads → the fp16 scaler spuriously skipped, which read
as flaky math but was a test-isolation bug (issue #58).
"""

import pytest

from src.data.download import dummy_texts
from src.data.tokenize import ByteTokenizer
from src.data.pack import pack_ids


@pytest.fixture(scope="session")
def toy_train_bin(tmp_path_factory):
    """A learnable byte-fallback corpus (ids < 256, fits toy vocab_size=256).

    Mirrors the documented offline pipeline (dummy_texts -> ByteTokenizer -> pack_ids).
    200 deterministic docs byte-encode to well over the ~tens-of-thousands of tokens the
    dt-bias loss-decrease test needs, and are structured/repetitive enough to learn.
    """
    tok = ByteTokenizer()
    ids = [t for doc in dummy_texts(200) for t in tok.encode(doc)]
    path = tmp_path_factory.mktemp("toydata") / "train.bin"
    pack_ids(ids, path)
    return path
