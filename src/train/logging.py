"""Minimal training logger: write JSON lines to a file and echo to stdout.

Backend-free. Plugs into `train.loop.train(logger=...)`; the loop hands it a
payload dict per logged step (step, lr, loss, grad_norm, val_loss, ...). Opens in
truncate mode by default (fresh run); pass `append=True` to preserve prior logs
across a resume.
"""

from __future__ import annotations

import json
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: str, echo: bool = True, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._f = self.path.open("a" if append else "w")

    def __call__(self, payload: dict) -> None:
        self._f.write(json.dumps(payload) + "\n")
        self._f.flush()
        if self.echo:
            keys = ("step", "lr", "loss", "grad_norm", "val_loss", "val_perplexity")
            parts = [f"{k}={payload[k]:.4g}" if isinstance(payload.get(k), float)
                     else f"{k}={payload[k]}" for k in keys if k in payload]
            print("  ".join(parts))

    def close(self) -> None:
        self._f.close()
