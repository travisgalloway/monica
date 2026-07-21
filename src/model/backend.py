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
    # Post-training (M9/#110) step factories, mirroring `make_train_step`. Implemented on
    # both the MLX and CUDA backends (the CUDA factories are torch mirrors, parity-tested).
    make_sft_train_step: Callable[..., Callable]
    make_dpo_train_step: Callable[..., Callable]
    make_grpo_train_step: Callable[..., Callable]
    # Distillation (M10): build a frozen, forward-only conversion teacher behind the seam
    # (`ConversionTeacher`). MLX-only for now; the CUDA branch raises NotImplementedError.
    make_teacher: Callable[..., Any]
    # Distillation (M10/#99): initialize a student model from a teacher (Mamba-in-the-Llama /
    # MOHAWK), returning an `InitReport`. MLX-only for now; CUDA raises NotImplementedError.
    init_student: Callable[..., Any]
    # Distillation (M10/#100): staged distill train-step factory (mixing-match / hidden-align /
    # logit-distill), mirroring make_*_train_step. MLX-only; CUDA raises NotImplementedError.
    make_distill_train_step: Callable[..., Callable]


def get_backend(name: str = "auto") -> Backend:
    """Return the `Backend` for `name` in {"auto", "mlx", "cuda"}.

    "auto" tries MLX (the Apple-Silicon dev backend) and falls back to CUDA/PyTorch.
    All backend imports happen inside the branch, so importing this module never pulls
    in a hardware library.
    """
    if name == "auto":
        # Decide by whether mlx is importable, NOT by catching SystemExit from
        # `_mlx_backend()` — that would also swallow an unrelated SystemExit raised
        # while mlx IS present and mis-route to the torch backend (a confusing
        # "torch not found" on an Apple-Silicon box).
        import importlib.util
        if importlib.util.find_spec("mlx") is not None:
            return _mlx_backend()
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

    def _make_teacher(config=None, *, pretrained=None, seed=0,
                      compute_dtype="fp32", compile=False):
        """Frozen conversion teacher (#93): `pretrained` (an HF checkpoint dir / repo id)
        loads real weights; otherwise a synthetic teacher is built from `config`.

        `compute_dtype` ("fp32" default; "fp16" a local Apple-Silicon opt-in that halves teacher
        memory) and `compile` (opt-in `mx.compile` of the logits-only forward) are MLX-local
        precompute levers — both default to the bit-identical eager fp32 path."""
        from .mlx_teacher import MLXConversionTeacher
        opts = dict(compute_dtype=compute_dtype, compile=compile)
        if pretrained is not None:
            return MLXConversionTeacher.from_pretrained(pretrained, config, **opts)
        if config is None:
            raise ValueError("make_teacher needs a TeacherConfig for the synthetic path "
                             "(pass `config=...`), or `pretrained=<dir/repo>` for real weights")
        return MLXConversionTeacher.from_config(config, seed=seed, **opts)

    def _init_student(student, teacher, method):
        """Initialize a student from a teacher (#99); `method` is an `InitMethod`."""
        from .mlx_student_init import init_student
        return init_student(student, teacher, method)

    def _make_distill_train_step(*args, **kwargs):
        from .mlx_distill import make_distill_train_step
        return make_distill_train_step(*args, **kwargs)

    def _make_optimizer(model, base_lr):
        # MLX AdamW holds no parameter refs at construction (params arrive at update);
        # `model` is read only for `.config.optimizer` (a uniform signature otherwise).
        # Muon (#237) is CUDA-only for now — raise loudly rather than silently falling
        # back to AdamW on a config that asked for Muon.
        if model.config.optimizer == "muon":
            raise NotImplementedError(
                "Muon is CUDA-only (#237); the MLX Newton-Schulz port is a scoped "
                "follow-up."
            )
        return optim.AdamW(learning_rate=base_lr)

    return Backend(
        name="mlx",
        model_cls=MLXMambaModel,
        make_train_step=make_train_step,
        save_optimizer=save_optimizer,
        load_optimizer=load_optimizer,
        make_optimizer=_make_optimizer,
        seed=lambda value: mx.random.seed(value),
        to_numpy=lambda a: np.array(a),
        make_sft_train_step=make_sft_train_step,
        make_dpo_train_step=_make_dpo_train_step,
        make_grpo_train_step=_make_grpo_train_step,
        make_teacher=_make_teacher,
        init_student=_init_student,
        make_distill_train_step=_make_distill_train_step,
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

    # Post-training (#110): the SFT/DPO/GRPO step factories now exist on the CUDA backend
    # too (torch mirrors of the MLX factories), imported lazily like make_train_step.
    def _make_sft_train_step(*args, **kwargs):
        from .cuda_train_step import make_sft_train_step
        return make_sft_train_step(*args, **kwargs)

    def _make_dpo_train_step(*args, **kwargs):
        from .cuda_train_step import make_dpo_train_step
        return make_dpo_train_step(*args, **kwargs)

    def _make_grpo_train_step(*args, **kwargs):
        from .cuda_train_step import make_grpo_train_step
        return make_grpo_train_step(*args, **kwargs)

    def _make_teacher(config=None, *, pretrained=None, seed=0,
                      compute_dtype="fp32", compile=False):
        """Frozen conversion teacher (#93/#94), torch port. `pretrained` (an HF checkpoint dir /
        repo id) loads real weights — this is how the dominant teacher precompute runs on the
        cloud GPU; otherwise a synthetic teacher is built from `config`.

        `compute_dtype`/`compile` are accepted for a uniform `make_teacher` signature but are
        MLX-local levers; the CUDA teacher's own dtype/torch.compile path is separate (#145)."""
        from .cuda_teacher import CUDATeacher
        if pretrained is not None:
            return CUDATeacher.from_pretrained(pretrained, config)
        if config is None:
            raise ValueError("make_teacher needs a TeacherConfig for the synthetic path "
                             "(pass `config=...`), or `pretrained=<dir/repo>` for real weights")
        return CUDATeacher.from_config(config, seed=seed)

    def _init_student(student, teacher, method):
        """Initialize a student from a teacher (#99), torch port; `method` is an `InitMethod`."""
        from .cuda_student_init import init_student
        return init_student(student, teacher, method)

    def _make_distill_train_step(*args, **kwargs):
        from .cuda_distill import make_distill_train_step
        return make_distill_train_step(*args, **kwargs)

    _dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _model_cls(cfg):
        return CUDAMambaModel(cfg, device=_dev)

    def _make_optimizer(model, base_lr):
        # Fused AdamW fuses the per-parameter update into one CUDA kernel — free throughput
        # on H100 (fewer kernel launches), no numerical change. It requires all params on a
        # CUDA device, so gate on it; on CPU (torch-CPU parity runs) fall back to the default.
        fused = _dev.startswith("cuda")
        cfg = model.config
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=base_lr, fused=fused)
        if cfg.optimizer == "muon":
            # Hybrid optimizer (#237): 2D hidden weight matrices go through Newton-Schulz
            # orthogonalization (Muon); everything else stays on AdamW. `is_muon_param` is
            # portable (src.model.blocks); the Muon/HybridOptimizer classes are torch-only
            # and stay lazily imported here, matching this module's lazy-import style.
            from .blocks import is_muon_param
            from .cuda_muon import Muon, HybridOptimizer
            muon_params, adam_params = [], []
            for name, p in model.named_parameters():
                (muon_params if is_muon_param(name, p.ndim) else adam_params).append(p)
            muon_lr = cfg.muon_lr if cfg.muon_lr is not None else base_lr
            adam = torch.optim.AdamW(adam_params, lr=base_lr, fused=fused) if adam_params else None
            muon = Muon(muon_params, lr=base_lr, lr_scale=muon_lr / base_lr,
                       momentum=cfg.muon_momentum, ns_steps=cfg.muon_ns_steps) if muon_params else None
            return HybridOptimizer(adam, muon)
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")

    return Backend(
        name="cuda",
        model_cls=_model_cls,
        make_train_step=_make_train_step,
        save_optimizer=_save_optimizer,
        load_optimizer=_load_optimizer,
        make_optimizer=_make_optimizer,
        seed=lambda value: torch.manual_seed(value),
        to_numpy=lambda a: a.detach().to("cpu").numpy(),
        make_sft_train_step=_make_sft_train_step,
        make_dpo_train_step=_make_dpo_train_step,
        make_grpo_train_step=_make_grpo_train_step,
        make_teacher=_make_teacher,
        init_student=_init_student,
        make_distill_train_step=_make_distill_train_step,
    )
