"""Backend-independent data pipeline.

Output is just integers on disk (uint16 memmap), written once and consumed
unchanged by every backend. No backend imports anywhere in this package.

Stages: download -> tokenize -> pack -> split -> loader. Packing/splitting/loading
operate on raw token-id arrays and never touch the tokenizer, so they are testable
offline with synthetic ids.
"""
