#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-/tmp/bb2m_indigo_charge_20261114/mixed_trainval}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/runs/indigo_charge_continue_20261114}"
INIT_JOINT="${INIT_JOINT:-$PROJECT_ROOT/runs/indigo_mixed_continue_20261113/best_joint.pt}"
ALIGNMENT_REPORT="${ALIGNMENT_REPORT:-$PROJECT_ROOT/runs/indigo_charge_alignment_v2_20261114/alignment_report.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m band_training.continue_indigo --dataset "$DATASET" --output "$OUTPUT" \
  --init-joint "$INIT_JOINT" --alignment-report "$ALIGNMENT_REPORT" \
  --epochs "${EPOCHS:-24}" --batch-size 8 --workers 6 --seed 20261114 \
  --charge-aware --box-lr 5e-5 --band-lr 1e-5 --backbone-lr 1e-6 --encoder-lr 2e-6 \
  --freeze-feature-epochs 1 --selection balanced --patience 6 --min-epochs 6 "$@"
