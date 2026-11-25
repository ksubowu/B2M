#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-/mnt/hdd2/bo/work/cvs/chemcv_data/real/indigo_style/mixed_trainval}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/runs/indigo_mixed_continue_20261113}"
INIT_RUN="${INIT_RUN:-$PROJECT_ROOT/runs/band_pilot_20261113}"
ALIGNMENT_REPORT="${ALIGNMENT_REPORT:-$PROJECT_ROOT/runs/indigo_alignment_20261113/alignment_report.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m band_training.continue_indigo --dataset "$DATASET" --output "$OUTPUT" \
  --init-run "$INIT_RUN" --alignment-report "$ALIGNMENT_REPORT" \
  --epochs "${EPOCHS:-3}" --batch-size "${BATCH_SIZE:-8}" --workers "${LOADER_WORKERS:-4}" "$@"
