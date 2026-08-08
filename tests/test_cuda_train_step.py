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

from src.model.blocks import MambaConfig, load_config
from src.model.cuda_backend import CUDAMambaModel
from src.model.cuda_train_step import (make_train_step, make_sft_train_step,
                                       make_dpo_train_step, make_grpo_train_step,
                                       save_optimizer, load_optimizer)
from src.train.loss_scale import DynamicLossScaler
from src.train.loop import TrainConfig, train
from src.train.checkpoint import CheckpointStore
from src.train.moe_balance import balancer_for_config, attach_balancer
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
    store = CheckpointStore(str(tmp_path / "resume"))
    store.save(step=half, loss_scale_state=None,
               weights_serializer=lambda p: ma.save(p),
               optimizer_serializer=lambda p: save_optimizer(oa, p))
    del ma, oa, sfa                                   # "kill"

    torch.manual_seed(999)
    mb = CUDAMambaModel(cfg)
    ob = _adam(mb)
    meta = store.load(weights_deserializer=lambda p: mb.load(p),
                      optimizer_deserializer=lambda p: load_optimizer(ob, p))
    sfb = make_train_step(mb, ob, grad_clip=1.0, scaler=None)
    run(mb, sfb, meta["step"], N, res)

    max_diff = max(abs(ref[s] - res[s]) for s in range(half, N))
    assert max_diff < 1e-4, f"resume not exact: max|diff|={max_diff:.3e}"


# --------------------------------------------------------------------------- #
# Loss-Free-Balancing threading (#213/#214): all four factories accept `balancer=`,
# the overflow-skip/pop ordering, and grad-accum count accumulation.
# --------------------------------------------------------------------------- #
def _moe_cfg(**over):
    base = dict(d_model=32, n_layers=2, head_dim=8, d_state=8, vocab_size=32,
                seq_len=16, precision="fp32", moe_every=1, n_experts=8, top_k=2,
                moe_d_ff=32, moe_balance_rate=0.05)
    base.update(over)
    return MambaConfig(**base)


def _pretrain_mb(cfg, B=4, L=16, seed=0):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, cfg.vocab_size, size=(B, L + 1))
    return tokens[:, :-1], tokens[:, 1:]


def _masked_mb(cfg, B=4, L=16, seed=0):
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, cfg.vocab_size, size=(B, L))
    tgt = rng.integers(0, cfg.vocab_size, size=(B, L))
    mask = np.ones((B, L), dtype=np.float32)
    return inp, tgt, mask


def test_make_train_step_accepts_balancer_and_emits_util_var():
    cfg = _moe_cfg()
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    balancer = balancer_for_config(cfg)
    step = make_train_step(model, _adam(model), grad_clip=1.0, scaler=None, balancer=balancer)
    attach_balancer(balancer, model)

    inp, tgt = _pretrain_mb(cfg)
    out = step(model, [(inp, tgt)], 1e-3)
    assert "moe_util_var" in out


def test_make_sft_train_step_accepts_balancer_and_emits_util_var():
    cfg = _moe_cfg()
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    balancer = balancer_for_config(cfg)
    step = make_sft_train_step(model, _adam(model), grad_clip=1.0, scaler=None,
                               balancer=balancer)
    attach_balancer(balancer, model)

    mb = _masked_mb(cfg)
    out = step(model, [mb], 1e-3)
    assert "moe_util_var" in out


def test_make_dpo_train_step_accepts_balancer_and_targets_policy_not_ref():
    cfg = _moe_cfg()
    torch.manual_seed(0)
    policy = CUDAMambaModel(cfg)
    torch.manual_seed(1)
    ref = CUDAMambaModel(cfg)
    balancer = balancer_for_config(cfg)
    step = make_dpo_train_step(policy, ref, _adam(policy), grad_clip=1.0, scaler=None,
                               balancer=balancer)
    attach_balancer(balancer, policy)

    c_in, c_tgt, c_mask = _masked_mb(cfg, seed=0)
    r_in, r_tgt, r_mask = _masked_mb(cfg, seed=1)
    out = step(policy, [(c_in, c_tgt, c_mask, r_in, r_tgt, r_mask)], 1e-3)
    assert "moe_util_var" in out
    # The balancer's pop/push must target the POLICY only -- the frozen reference is
    # never wired to it, so its bias stays exactly whatever it started with (inactive).
    assert ref.moe_biases() == [[], []]
    assert policy.moe_biases() != [[], []]


def test_make_grpo_train_step_accepts_balancer_and_emits_util_var():
    cfg = _moe_cfg()
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    balancer = balancer_for_config(cfg)
    step = make_grpo_train_step(model, _adam(model), grad_clip=1.0, scaler=None,
                                balancer=balancer)
    attach_balancer(balancer, model)

    inp, tgt, mask = _masked_mb(cfg)
    adv = np.random.default_rng(0).normal(size=(inp.shape[0],)).astype(np.float32)
    out = step(model, [(inp, tgt, mask, adv)], 1e-3)
    assert "moe_util_var" in out


def test_overflow_skip_does_not_pop_moe_load_counts():
    """The balancer hook runs AFTER the fp16 overflow early-return, so a skipped step's
    routed-token counts are NOT popped -- they must survive into the next pop
    (mirrors `mlx_backend.py:717-720`'s documented, intentional behavior)."""
    cfg = _moe_cfg()
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    balancer = balancer_for_config(cfg)
    # A scale above fp32-max forces loss*scale -> inf -> non-finite grads (robust trigger,
    # same recipe as test_fp16_overflow_skips_update_and_backs_off above).
    scaler = DynamicLossScaler(init_scale=1e40, backoff=0.5, min_scale=1.0)
    step = make_train_step(model, _adam(model), grad_clip=1.0, scaler=scaler,
                           balancer=balancer)
    attach_balancer(balancer, model)          # turns load counting ON

    inp, tgt = _pretrain_mb(cfg)
    out = step(model, [(inp, tgt)], 1e-3)
    assert out["skipped"] is True
    assert "moe_util_var" not in out

    # The forward pass ran (and counted) before overflow was detected in backward; since
    # the balancer hook never ran, those counts were never popped -- a fresh pop must
    # still see them.
    survived = model.pop_moe_load()
    assert sum(sum(layer) for layer in survived) > 0


def test_grad_accum_two_microbatches_accumulates_moe_load_counts():
    """Grad accumulation sums counts across BOTH micro-batches, not just the last one."""
    cfg = _moe_cfg(moe_balance_rate=None)     # counting only, no balancer consuming it
    inp, tgt = _pretrain_mb(cfg)

    torch.manual_seed(0)
    model1 = CUDAMambaModel(cfg)
    step1 = make_train_step(model1, _adam(model1), grad_clip=1.0, scaler=None, balancer=None)
    model1.set_moe_load_counting(True)
    step1(model1, [(inp, tgt)], 1e-3)
    single = model1.pop_moe_load()

    torch.manual_seed(0)                      # identical fresh weights, pre-optimizer-step
    model2 = CUDAMambaModel(cfg)
    step2 = make_train_step(model2, _adam(model2), grad_clip=1.0, scaler=None, balancer=None)
    model2.set_moe_load_counting(True)
    step2(model2, [(inp, tgt), (inp, tgt)], 1e-3)
    double = model2.pop_moe_load()

    assert sum(sum(l) for l in single) > 0     # sanity: routing actually happened
    for s_layer, d_layer in zip(single, double):
        assert d_layer == [2 * c for c in s_layer]
