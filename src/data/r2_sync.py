"""Sync a locally-built artifact tree to/from object storage (#80).

The corpus/teacher/checkpoint builders all write **local directory trees** (cleaned Parquet,
tokenized `.bin`/`.bounds`/`manifest.json`, ...). This module is the thin, isolated seam that
pushes one of those trees to a durable S3-compatible store — **Cloudflare R2** for us — and pulls
it back on a compute host. It is the runbook's "build locally, then sync results to durable
storage" step (docs/infrastructure.md): nothing in the hot data path changes; the few large
shards are simply mirrored under the same `src/data/storage.py` prefix on R2.

Addressing is storage-URI agnostic (the `corpus.py` writers already are): `file://` / bare paths
resolve to the local FS, `s3://bucket/prefix` resolves to R2 via `s3fs`. R2 is not AWS-default, so
its **endpoint** is injected from the environment (`AWS_ENDPOINT_URL_S3`); the key/secret are read
by `botocore`/`s3fs` from `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (never committed — see
`.env.example`).

ABOVE THE SEAM — no `mlx`/`torch`. `fsspec`/`s3fs` are imported LAZILY inside the functions, so
importing this module stays cheap and the seam guard (tests/test_import_guard.py) needs nothing
extra.

CLI:
    set -a; . ./.env; set +a                          # load HF/R2 secrets into the env
    python -m src.data.r2_sync up   data/poc-distill   s3://monica/poc-distill
    python -m src.data.r2_sync down s3://monica/poc-distill   data/poc-distill
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

#: Environment variables that hold the R2 S3 endpoint, in precedence order. (`s3fs` reads the
#: key/secret itself from the standard `AWS_*` vars; only the non-AWS endpoint must be injected.)
_ENDPOINT_ENV = ("AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL", "R2_ENDPOINT")


def r2_endpoint() -> str | None:
    """The configured R2/S3 endpoint URL from the environment, or None (then s3fs hits AWS)."""
    for var in _ENDPOINT_ENV:
        val = os.environ.get(var)
        if val:
            return val
    return None


def _fs_for(uri):
    """Resolve an fsspec URI to (filesystem, path). For `s3://` inject the R2 endpoint from the
    env; everything else (bare paths, `file://`, `memory://`) resolves with no extra options."""
    import fsspec

    opts: dict = {}
    if str(uri).startswith("s3://"):
        endpoint = r2_endpoint()
        if endpoint:
            opts["client_kwargs"] = {"endpoint_url": endpoint}
    return fsspec.core.url_to_fs(str(uri), **opts)


def _relpath(remote: str, root: str) -> str:
    """`remote` with the `root` prefix stripped, as a clean relative posix path."""
    rel = remote[len(root):] if remote.startswith(root) else remote
    return rel.lstrip("/")


def upload_dir(local_dir, dst_uri) -> List[str]:
    """Mirror every file under local `local_dir` to `dst_uri` (a directory prefix on any fsspec
    backend), preserving the relative tree. Returns the destination paths written, in order.

    Per-file `put_file` (not a recursive `put`) keeps the behaviour identical across the local /
    `memory://` / `s3://` backends the tests and R2 use."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise NotADirectoryError(f"{local_dir} is not a directory")
    fs, root = _fs_for(dst_uri)
    root = root.rstrip("/")
    written: List[str] = []
    for p in sorted(local_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir).as_posix()
        remote = f"{root}/{rel}"
        fs.makedirs(remote.rsplit("/", 1)[0], exist_ok=True)
        fs.put_file(str(p), remote)
        written.append(remote)
    return written


def download_dir(src_uri, local_dir) -> List[str]:
    """Pull every file under `src_uri` (a directory prefix on any fsspec backend) to local
    `local_dir`, preserving the relative tree. Reverses `upload_dir`; returns the local paths."""
    fs, root = _fs_for(src_uri)
    root = root.rstrip("/")
    local_dir = Path(local_dir)
    got: List[str] = []
    for remote in sorted(fs.find(root)):           # files only, recursive
        dst = local_dir / _relpath(remote, root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(remote, str(dst))
        got.append(str(dst))
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up", help="upload a local dir tree to an fsspec URI (e.g. s3://...)")
    up.add_argument("local")
    up.add_argument("dst")
    dn = sub.add_parser("down", help="download an fsspec URI tree to a local dir")
    dn.add_argument("src")
    dn.add_argument("local")
    args = ap.parse_args()

    if args.cmd == "up":
        written = upload_dir(args.local, args.dst)
        print(f"uploaded {len(written)} file(s): {args.local} -> {args.dst}")
    else:
        got = download_dir(args.src, args.local)
        print(f"downloaded {len(got)} file(s): {args.src} -> {args.local}")


if __name__ == "__main__":
    main()
