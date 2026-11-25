#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-/mnt/hdd2/bo/work/cvs/pubchem/band_train_pilot_20261113}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/runs/band_pilot_$(date -u +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec "$PYTHON_BIN" -m band_training.train --dataset "$DATASET" --output "$OUTPUT" \
  --tiny-steps "${TINY_STEPS:-300}" --steps "${BAND_STEPS:-1800}" \
  --atom-steps "${ATOM_STEPS:-300}" --batch-size "${BATCH_SIZE:-8}"
