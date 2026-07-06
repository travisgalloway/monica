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

echo "[A1] fused Triton SSD scan engages via mamba_ssm.ops.triton (the critical path)"
python -c "
# Test the actual code path, not the top-level mamba_ssm import.
# mamba_ssm.__init__ imports selective_scan_cuda which has a known ABI mismatch on
# torch 2.12+cu130 (symbol 'ib' vs 'jb' for unsigned int). The _fused_scan() stub
# bypasses __init__ and imports ops.triton.ssd_combined directly — that path works.
# causal_conv1d has the same ABI issue; the code falls back to PyTorch conv (minor hit).
from src.model.cuda_backend import _fused_scan
s = _fused_scan()
assert s is not None, 'fused scan kernel not loaded — mamba-ssm install or GPU issue'
print('  fused scan kernel OK:', s.__module__)
import sys; sys.path.insert(0, '.')
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
print('  mamba_ssm.ops.triton import OK')
import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda)
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

echo "[A6] /vol volume capacity >= 2 TB free (combined ≈1.57 TB cache + append/merge scratch, #177)"
_VOL_FREE_KB=$(df -Pk /vol | tail -1 | awk '{print $4}')
_VOL_FLOOR_KB=$((2 * 1024 * 1024 * 1024))
if [ "$_VOL_FREE_KB" -lt "$_VOL_FLOOR_KB" ]; then
  echo "FAIL: /vol has only $((_VOL_FREE_KB / 1024 / 1024)) GB free, need >= 2048 GB (2 TB) — see docs/runbooks/m10-phase-bprime-append.md Prereqs"
  exit 1
fi
echo "  /vol free: $((_VOL_FREE_KB / 1024 / 1024)) GB (>= 2048 GB floor) OK"

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

echo "[B3] assert: dtype=uint32, .bounds present, max token id < 151669 (Qwen3 effective vocab), train/val provably disjoint"
python -c "
import json, numpy as np
from pathlib import Path

corpus_man = json.loads((Path('/vol/corpus8k') / 'manifest.json').read_text())
corpus_total = int(corpus_man['n_tokens'])

split_total = 0
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
    split_total += int(meta['n_tokens'])

# Disjointness: split_shards() (src/data/split.py) holds out the last val_tokens as a
# contiguous tail and concatenates everything before it into train -- disjoint by
# construction PROVIDED nothing was double-counted or silently dropped. The cheapest
# correct on-pod check is the token-count identity: train + val must account for exactly
# the whole corpus, no more, no less.
assert split_total == corpus_total, (
    f'train+val tokens ({split_total:,}) != corpus tokens ({corpus_total:,}) -- '
    'the split is not a clean disjoint partition of the corpus'
)
print(f'  train+val == corpus total ({corpus_total:,} tokens): disjoint OK')

# doc-boundary .bounds sidecars are required for the SSM state-reset (#68)
bounds = list(Path('/vol/corpus8k').glob('*.bounds'))
assert bounds, 'no .bounds sidecars in /vol/corpus8k — doc-boundary reset (#68) broken'
print(f'  .bounds: {len(bounds)} shards ok')
"

echo "[B4] append-input integrity (#177): existence/meta only, no full download"
python -c "
from src.data.r2_sync import _fs_for

def check(prefix, label):
    uri = f's3://monica-training/{prefix}'
    fs, root = _fs_for(uri)
    root = root.rstrip('/')
    # non-recursive top-level listing -- fs.find() would recursively walk every object under
    # the prefix (slow + many LIST calls on the 566 GB base teacher cache); existence only needs
    # one level.
    entries = fs.ls(root, detail=False)
    assert entries, f'{label}: no entries found under {uri} — #177 append cannot proceed'
    manifest = f'{root}/manifest.json'
    assert fs.exists(manifest), f'{label}: manifest.json missing at {manifest}'
    size = fs.info(manifest)['size']
    print(f'  {label}: {len(entries)} top-level entries under {uri}, manifest.json {size} bytes OK')

# the corpus extension (#176/#182) — append_new_chunks.py's --extension-shards source
check('poc-distill-ext/corpus/tokenized/qwen3-8k', 'extension corpus')
# the frozen base teacher cache — append_new_chunks.py's --topk-dir / shard-0 source
check('poc-distill/teacher-outputs/topk-logits-merged', 'frozen base teacher cache')
"

echo "── B PASS $(date) ──"

# ─── C. CUDA smoke (before the dominant-cost precompute and sweep) ────────────
echo ""
echo "── C. CUDA smoke $(date) ──"

echo "[C0] CUDA availability gate (fail here, not 20 min into smoke)"
python -c "
import sys, torch
if not torch.cuda.is_available():
    print(f'CUDA NOT AVAILABLE: torch {torch.__version__} (built for cu{(torch.version.cuda or \"\").replace(\".\",\"\")}) is incompatible with this machine driver.', file=sys.stderr)
    print('Run bootstrap.sh again — it will detect and reinstall torch+cu124.', file=sys.stderr)
    sys.exit(1)
name = torch.cuda.get_device_name(0)
mem_gb = torch.cuda.get_device_properties(0).total_memory // 1024**3
print(f'  CUDA OK: {name} ({mem_gb} GB) | torch {torch.__version__} cu{(torch.version.cuda or \"\").replace(\".\",\"\")}')
"

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

echo "[C2b] precompute_teacher REAL (non-synthetic) tiny slice — confirms Qwen3-4B-Thinking-2507"
echo "      itself loads on CUDA (per-head QK RMSNorm, no QKV bias, rope_theta 5e6) BEFORE the"
echo "      dominant-cost precompute spend. [C2] above only proves the synthetic cache-write path."
# Real qwen3 tokenizer (network, cheap) + 2 tiny chunks at seq_len=32 -- small enough to be
# pennies on the already-rented pod, but a real (not synthetic) teacher.topk_logits() forward.
python -c "
from pathlib import Path
from src.data.corpus import ingest_dummy
from src.data.distill_corpus import build_distill_corpus
from src.data.split import split_shards
from src.data import storage
out = Path('/tmp/m10-real-teacher-smoke')
build_distill_corpus(ingest_dummy(400, source='smoke'), out, tokenizer='qwen3',
                     seq_len=32, byte_fallback=False)
tok_dir = storage.corpus_tokenized_dir(out, 'qwen3', 32)
split_shards(tok_dir, Path('/tmp/m10-real-teacher-smoke/split'), val_tokens=0)
print('tiny real-tokenizer split ready (seq_len=32, train only)')
"
cat > /tmp/m10-real-teacher-manifest.yaml <<'YAML'
# Minimal manifest for the preflight real-teacher probe only -- precompute_teacher.py reads just
# conversion_teacher + seq_len from it (layout/stages/init are unused on this path).
student: preflight-real-teacher-probe
conversion_teacher: Qwen/Qwen3-4B-Thinking-2507
tokenizer: qwen3
seq_len: 32
init: mohawk
stages: [logit-distill]
layout: {}
YAML
python scripts/precompute_teacher.py \
  --manifest /tmp/m10-real-teacher-manifest.yaml \
  --data /tmp/m10-real-teacher-smoke/split --splits train \
  --out /tmp/m10-real-teacher-smoke/teacher-outputs \
  --backend cuda --k 8 --batch-size 2
echo "  real Qwen3-4B-Thinking-2507 teacher forward OK"

echo "[C3] pytest tests/test_cuda_parity.py (torch-backend SSM + forward/step parity)"
python -m pytest tests/test_cuda_parity.py -q

echo "── C PASS $(date) ──"

echo ""
echo "=== M10 preflight: ALL CHECKS PASSED $(date) ==="
