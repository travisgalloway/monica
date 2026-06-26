#!/bin/bash
# M10 pod preflight — runs runbook verification checklist sections A, B, C before any GPU spend.
# Ordered cheapest-first so failures surface before the dominant precompute cost.
# Idempotent: safe to re-run. Exit 0 = all checks pass; set -e exits 1 on the first failure.
set -euo pipefail

cd /workspace/monica
set -a; . /workspace/monica/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== M10 preflight: $(date) ==="

# ─── A. Environment (before any GPU spend) ────────────────────────────────────
echo ""
echo "── A. Environment $(date) ──"

echo "[A1] mamba_ssm + causal_conv1d importable (cuda-fast succeeded)"
python -c "
import mamba_ssm, causal_conv1d
print('  mamba_ssm', mamba_ssm.__version__, 'causal_conv1d', causal_conv1d.__version__)
"

echo "[A2] fused Triton SSD scan engages (not the silent pure-PyTorch fallback)"
python -c "
from src.model.cuda_backend import _fused_scan
s = _fused_scan()
assert s is not None, 'fused scan kernel not loaded — cuda-fast install failed or GPU not visible'
print('  fused scan kernel:', s)
"

echo "[A3] s3fs==2026.2.0 pin holds"
python -c "
import s3fs
assert s3fs.__version__ == '2026.2.0', f'got {s3fs.__version__} (bare pip install upgrades fsspec and breaks datasets)'
print('  s3fs', s3fs.__version__)
"

echo "[A4] r2_sync round-trip (.env creds, R2 in-region)"
_SMOKE=$(mktemp -d)
printf 'm10-preflight-probe\n' > "$_SMOKE/probe.txt"
python -m src.data.r2_sync up "$_SMOKE" s3://monica-training/_smoke/m10-preflight
_SMOKE_DN=$(mktemp -d)
python -m src.data.r2_sync down s3://monica-training/_smoke/m10-preflight "$_SMOKE_DN"
diff "$_SMOKE/probe.txt" "$_SMOKE_DN/probe.txt"
python -c "
from src.data.r2_sync import _fs_for
fs, _ = _fs_for('s3://monica-training/')
[fs.rm_file(f) for f in fs.find('monica-training/_smoke/m10-preflight')]
"
echo "  r2_sync round-trip OK"

echo "[A5] Ampere+ card required for bf16 at seq 8192 (A100/H100)"
nvidia-smi | grep -E "A100|H100" || { echo "FAIL: expected A100 or H100 in nvidia-smi — bf16 at seq 8192 needs Ampere+"; exit 1; }

echo "── A PASS $(date) ──"

# ─── B. Corpus integrity (cheap — before the dominant-cost precompute) ────────
echo ""
echo "── B. Corpus integrity $(date) ──"

echo "[B1] r2_sync down qwen3-8k tokenized corpus"
mkdir -p /vol/corpus8k
python -m src.data.r2_sync down \
  s3://monica-training/poc-distill/corpus/tokenized/qwen3-8k /vol/corpus8k

echo "[B2] split --shards -> train.bin / val.bin"
python -m src.data.split \
  --shards /vol/corpus8k --out /vol/split8k --val-tokens 10000000

echo "[B3] assert: dtype=uint32, .bounds present, max token id < 151669 (Qwen3 effective vocab)"
python -c "
import json, numpy as np
from pathlib import Path

for split in ('train', 'val'):
    meta = json.loads((Path('/vol/split8k') / f'{split}.meta.json').read_text())
    assert meta['dtype'] == 'uint32', f'{split} dtype={meta[\"dtype\"]} (expected uint32)'
    # memmap avoids loading the full 7.5 GB train shard into RAM
    data = np.memmap(f'/vol/split8k/{split}.bin', dtype=np.uint32, mode='r')
    max_id = int(data.max())
    assert max_id < 151669, (
        f'{split} max token id {max_id} >= 151669 (Qwen3 effective vocab); '
        f'the vocab-bound check from #153/#154 may not have been applied at build time'
    )
    print(f'  {split}: dtype=uint32  n_tokens={meta[\"n_tokens\"]:,}  max_id={max_id}')

# doc-boundary .bounds sidecars are required for the SSM state-reset (#68)
bounds = list(Path('/vol/corpus8k').glob('*.bounds'))
assert bounds, 'no .bounds sidecars in /vol/corpus8k — doc-boundary reset (#68) broken'
print(f'  .bounds: {len(bounds)} shards ok')
"

echo "── B PASS $(date) ──"

# ─── C. CUDA smoke (before the dominant-cost precompute and sweep) ────────────
echo ""
echo "── C. CUDA smoke $(date) ──"

echo "[C1] distill_smoke --backend cuda (student init + all 3 distill stages)"
python scripts/distill_smoke.py --backend cuda

echo "[C2] precompute_teacher --synthetic --backend cuda (teacher load + top-k cache write)"
# Build a fresh tiny toy split so the check is self-contained
python -c "
from pathlib import Path
from src.data.corpus import ingest_dummy
from src.data.distill_corpus import build_distill_corpus
from src.data.split import split_shards
from src.data import storage
out = Path('/tmp/m10-smoke')
build_distill_corpus(ingest_dummy(400, source='smoke'), out, tokenizer='qwen3',
                     seq_len=16, byte_fallback=True)
tok_dir = storage.corpus_tokenized_dir(out, 'qwen3', 16)
split_shards(tok_dir, Path('/tmp/m10-smoke/split'), val_tokens=16 * 4)
print('toy split ready')
"
python scripts/precompute_teacher.py \
  --manifest config/toy-distill.yaml \
  --data /tmp/m10-smoke/split \
  --out /tmp/m10-smoke/teacher-outputs \
  --backend cuda --k 8 --synthetic

echo "[C3] pytest tests/test_cuda_parity.py (torch-backend SSM + forward/step parity)"
python -m pytest tests/test_cuda_parity.py -q

echo "── C PASS $(date) ──"

echo ""
echo "=== M10 preflight: ALL CHECKS PASSED $(date) ==="
