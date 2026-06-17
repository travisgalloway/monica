"""Backend selection — the portable factory above the seam.

The driver scripts (`scripts/train.py`, `scripts/smoke_test.py`) must run on either
Apple Silicon (MLX) or a CUDA host (PyTorch) without code changes. The `ModelInterface`
seam already isolates the *model*; this module isolates the rest of the backend wiring
the drivers need — the optimizer, the injected `train_step`, the resume-bundle
(de)serializers, RNG seeding, and the array->numpy converter — behind one
`get_backend(name)` call.

THIS MODULE MUST NOT IMPORT A BACKEND AT MODULE LEVEL (no `mlx`, no `torch`). Every
backend import lives inside a `get_backend` branch (lazy), so the module is importable
on any host and stays in `tests/test_import_guard.py`'s portable set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Backend:
    """Everything a driver needs to run on one hardware backend.

    `model_cls(config)` builds a `ModelInterface`; `make_optimizer(model, base_lr)`
    builds its optimizer. The remaining callables mirror the backend's `*_train_step`
    module so the drivers never import `mlx`/`torch` directly.
    """

    name: str
    model_cls: type
    make_train_step: Callable[..., Callable]
    save_optimizer: Callable[[Any, str], None]
    load_optimizer: Callable[[Any, str], None]
    make_optimizer: Callable[[Any, float], Any]
    seed: Callable[[int], None]
    to_numpy: Callable[[Any], Any]
    # Post-training (M9) step factories, mirroring `make_train_step`. SFT/DPO are
    # MLX-only for now; the CUDA branch raises a clear NotImplementedError.
    make_sft_train_step: Callable[..., Callable]
    make_dpo_train_step: Callable[..., Callable]
    make_grpo_train_step: Callable[..., Callable]
    # Distillation (M10): build a frozen, forward-only conversion teacher behind the seam
    # (`ConversionTeacher`). MLX-only for now; the CUDA branch raises NotImplementedError.
    make_teacher: Callable[..., Any]


def get_backend(name: str = "auto") -> Backend:
    """Return the `Backend` for `name` in {"auto", "mlx", "cuda"}.

    "auto" tries MLX (the Apple-Silicon dev backend) and falls back to CUDA/PyTorch.
    All backend imports happen inside the branch, so importing this module never pulls
    in a hardware library.
    """
    if name == "auto":
        try:
            return _mlx_backend()
        except SystemExit:
            # mlx absent on this host — fall back to the torch backend.
            return _cuda_backend()
    if name == "mlx":
        return _mlx_backend()
    if name == "cuda":
        return _cuda_backend()
    raise ValueError(f"unknown backend {name!r}; expected one of auto, mlx, cuda")


def _mlx_backend() -> Backend:
    try:
        import mlx.core as mx
        import mlx.optimizers as optim
    except ModuleNotFoundError as e:
        if e.name != "mlx":
            raise
        raise SystemExit(
            "mlx not found — run with the project venv on Apple Silicon:\n"
            "    .venv/bin/python scripts/<driver>.py ...\n"
            "(mlx installs only on Apple Silicon via the '[mlx]' extra; a bare "
            "`python` likely points at a different interpreter.)"
        ) from e
    import numpy as np

    from .mlx_backend import MLXMambaModel
    from .mlx_train_step import (make_train_step, make_sft_train_step,
                                 save_optimizer, load_optimizer)

    # Lazy so constructing the backend never requires the DPO/GRPO step to exist yet.
    def _make_dpo_train_step(*args, **kwargs):
        from .mlx_train_step import make_dpo_train_step
        return make_dpo_train_step(*args, **kwargs)

    def _make_grpo_train_step(*args, **kwargs):
        from .mlx_train_step import make_grpo_train_step
        return make_grpo_train_step(*args, **kwargs)

    def _make_teacher(config=None, *, pretrained=None, seed=0):
        """Frozen conversion teacher (#93): `pretrained` (an HF checkpoint dir / repo id)
        loads real weights; otherwise a synthetic teacher is built from `config`."""
        from .mlx_teacher import MLXConversionTeacher
        if pretrained is not None:
            return MLXConversionTeacher.from_pretrained(pretrained, config)
        if config is None:
            raise ValueError("make_teacher needs a TeacherConfig for the synthetic path "
                             "(pass `config=...`), or `pretrained=<dir/repo>` for real weights")
        return MLXConversionTeacher.from_config(config, seed=seed)

    return Backend(
        name="mlx",
        model_cls=MLXMambaModel,
        make_train_step=make_train_step,
        save_optimizer=save_optimizer,
        load_optimizer=load_optimizer,
        # MLX AdamW holds no parameter refs at construction (params arrive at update);
        # `model` is accepted for a uniform signature and ignored.
        make_optimizer=lambda model, base_lr: optim.AdamW(learning_rate=base_lr),
        seed=lambda value: mx.random.seed(value),
        to_numpy=lambda a: np.array(a),
        make_sft_train_step=make_sft_train_step,
        make_dpo_train_step=_make_dpo_train_step,
        make_grpo_train_step=_make_grpo_train_step,
        make_teacher=_make_teacher,
    )


def _cuda_backend() -> Backend:
    try:
        import torch
    except ModuleNotFoundError as e:
        if e.name != "torch":
            raise
        raise SystemExit(
            "torch not found — install the CUDA backend extra on a Linux/CUDA host:\n"
            "    pip install -e '.[cuda]'\n"
            "(torch is omitted from the default deps; mlx is the Apple-Silicon backend.)"
        ) from e
    import numpy as np

    from .cuda_backend import CUDAMambaModel

    # The train-step primitives land with the CUDA train_step issue (#37); import them
    # lazily so the `cuda` branch is usable for model construction (which raises the
    # stub's clear NotImplementedError) before they exist.
    def _make_train_step(*args, **kwargs):
        from .cuda_train_step import make_train_step
        return make_train_step(*args, **kwargs)

    def _save_optimizer(optimizer, path):
        from .cuda_train_step import save_optimizer
        return save_optimizer(optimizer, path)

    def _load_optimizer(optimizer, path):
        from .cuda_train_step import load_optimizer
        return load_optimizer(optimizer, path)

    def _post_training_unsupported(*args, **kwargs):
        raise NotImplementedError(
            "SFT/DPO (M9) are implemented on the MLX dev backend only; the CUDA "
            "post-training steps are deferred (run scripts/sft.py / scripts/dpo.py on "
            "Apple Silicon).")

    def _teacher_unsupported(*args, **kwargs):
        raise NotImplementedError(
            "The conversion teacher (M10/#93) is implemented on the MLX dev backend only; "
            "the CUDA teacher loader is deferred.")

    return Backend(
        name="cuda",
        model_cls=CUDAMambaModel,
        make_train_step=_make_train_step,
        save_optimizer=_save_optimizer,
        load_optimizer=_load_optimizer,
        make_optimizer=lambda model, base_lr: torch.optim.AdamW(
            model.parameters(), lr=base_lr),
        seed=lambda value: torch.manual_seed(value),
        to_numpy=lambda a: a.detach().to("cpu").numpy(),
        make_sft_train_step=_post_training_unsupported,
        make_dpo_train_step=_post_training_unsupported,
        make_grpo_train_step=_post_training_unsupported,
        make_teacher=_teacher_unsupported,
    )
