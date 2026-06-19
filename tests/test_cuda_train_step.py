"""CUDA/PyTorch train_step: grad accumulation, fp16 loss scaling, loop + resume.

Skipped where torch is unavailable (runs on CPU — no GPU needed). Mirrors
tests/test_mlx_train_step.py and adds an end-to-end check that the unchanged portable
train loop drives the torch backend (loss decreases) and that save/kill/resume is exact
in fp32 — the latter via the M4 smoke-gate's fixed-batch protocol (the loop's shuffled
infinite stream is not position-resumable by design; the smoke gate bypasses it for the
same reason).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.blocks import load_config
from src.model.cuda_backend import CUDAMambaModel
from src.model.cuda_train_step import make_train_step, save_optimizer, load_optimizer
from src.train.loss_scale import DynamicLossScaler
from src.train.loop import TrainConfig, train
from src.train.checkpoint import save_resume, load_resume
from src.data.pack import pack_ids
from src.data.loader import PackedLoader

TOY_CFG = "config/toy.yaml"


def _rand_batch(cfg, B=4, L=32, seed=0):
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    tgt = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    return inp, tgt


def _adam(model, lr=1e-3):
    return torch.optim.AdamW(model.parameters(), lr=lr)


def _packed_loader(cfg, tmp_path, n_tokens, batch_size=4, shuffle=True, seed=0):
    ids = np.random.default_rng(0).integers(0, cfg.vocab_size, size=n_tokens)
    path = tmp_path / "train.bin"
    pack_ids(ids, path)
    return PackedLoader(path, cfg.seq_len, batch_size, shuffle=shuffle, seed=seed)


def test_grad_accum_two_identical_microbatches_equal_single():
    cfg = load_config(TOY_CFG)
    inp, tgt = _rand_batch(cfg)

    torch.manual_seed(0)
    m1 = CUDAMambaModel(cfg)
    r1 = make_train_step(m1, _adam(m1), grad_clip=1.0, scaler=None)(m1, [(inp, tgt)], 1e-3)

    torch.manual_seed(0)
    m2 = CUDAMambaModel(cfg)
    r2 = make_train_step(m2, _adam(m2), grad_clip=1.0)(m2, [(inp, tgt), (inp, tgt)], 1e-3)

    # Averaging two identical micro-batches == one micro-batch.
    assert abs(r1["loss"] - r2["loss"]) < 1e-4
    assert abs(r1["grad_norm"] - r2["grad_norm"]) < 1e-4


def test_fp16_overflow_skips_update_and_backs_off():
    cfg = load_config(TOY_CFG)
    inp, tgt = _rand_batch(cfg)

    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    opt = _adam(model)
    # A scale above fp32-max makes loss*scale -> inf, so the gradients are non-finite
    # regardless of per-element magnitude (robust trigger).
    scaler = DynamicLossScaler(init_scale=1e40, backoff=0.5, min_scale=1.0)
    step_fn = make_train_step(model, opt, grad_clip=1.0, scaler=scaler)

    before = [p.detach().clone() for p in model.parameters()]
    out = step_fn(model, [(inp, tgt)], 1e-3)
    after = list(model.parameters())

    assert out["skipped"] is True
    assert scaler.scale == 0.5e40                 # backed off on overflow
    for b, a in zip(before, after):
        assert torch.equal(b, a)                  # weights untouched on a skipped step


def test_fp16_clean_step_updates_and_reports_scale():
    cfg = load_config(TOY_CFG)
    inp, tgt = _rand_batch(cfg)

    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    scaler = DynamicLossScaler(init_scale=1024.0)
    step_fn = make_train_step(model, _adam(model), grad_clip=1.0, scaler=scaler)

    out = step_fn(model, [(inp, tgt)], 1e-3)
    assert out["skipped"] is False
    assert out["loss_scale"] == 1024.0
    assert np.isfinite(out["grad_norm"])


def test_loop_loss_decreases(tmp_path):
    """The unchanged portable train loop drives the torch backend; loss goes down."""
    cfg = load_config(TOY_CFG)
    loader = _packed_loader(cfg, tmp_path, n_tokens=70 * (cfg.seq_len + 1))

    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    model.train()
    step_fn = make_train_step(model, _adam(model, lr=3e-3), grad_clip=1.0, scaler=None)

    losses = []
    tcfg = TrainConfig(total_steps=60, base_lr=3e-3, warmup_steps=5, grad_accum=1,
                       log_every=1, eval_every=10 ** 9, ckpt_every=10 ** 9)
    train(model, loader, tcfg, step_fn, logger=lambda p: losses.append(p["loss"]))

    assert np.mean(losses[-5:]) < np.mean(losses[:5]), losses


def test_save_kill_resume_exact(tmp_path):
    """fp32 save/kill/resume is exact: rebuild + load weights + optimizer + step, and
    the post-resume loss trajectory matches the uninterrupted reference (atol 1e-4)."""
    cfg = load_config(TOY_CFG)
    assert cfg.precision == "fp32"
    N, half = 8, 4

    # Fixed batch stream (shuffle off) so batch at step s is identical in both runs.
    loader = _packed_loader(cfg, tmp_path, n_tokens=(N + 2) * 4 * (cfg.seq_len + 1),
                            batch_size=4, shuffle=False)
    batches = []
    for b in loader.epoch():
        batches.append(b)
        if len(batches) == N:
            break
    assert len(batches) == N

    def fresh():
        torch.manual_seed(0)
        model = CUDAMambaModel(cfg)
        opt = _adam(model)
        return model, opt, make_train_step(model, opt, grad_clip=1.0, scaler=None)

    def run(model, step_fn, lo, hi, into):
        for s in range(lo, hi):
            into[s] = step_fn(model, [batches[s]], 1e-3)["loss"]

    ref = {}
    m, _, sf = fresh()
    run(m, sf, 0, N, ref)

    res = {}
    ma, oa, sfa = fresh()
    run(ma, sfa, 0, half, res)
    weights = str(tmp_path / "weights.safetensors")
    bundle = str(tmp_path / "resume")
    ma.save(weights)
    save_resume(bundle, step=half, loss_scale_state=None,
                optimizer_serializer=lambda p: save_optimizer(oa, p))
    del ma, oa, sfa                                   # "kill"

    torch.manual_seed(999)
    mb = CUDAMambaModel(cfg)
    mb.load(weights)
    ob = _adam(mb)
    meta = load_resume(bundle, optimizer_deserializer=lambda p: load_optimizer(ob, p))
    sfb = make_train_step(mb, ob, grad_clip=1.0, scaler=None)
    run(mb, sfb, meta["step"], N, res)

    max_diff = max(abs(ref[s] - res[s]) for s in range(half, N))
    assert max_diff < 1e-4, f"resume not exact: max|diff|={max_diff:.3e}"
