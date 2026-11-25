#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${BB2M_GPU:-0}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "${PYTHON_BIN:-python}" -m band_training.train_scored_bonds \
  --dataset /mnt/hdd2/bo/work/cvs/chemcv_data/real/indigo_full_v3_20261116 \
  --init runs/full_v3_paired_formal_20261117/best_joint.pt \
  --target-audit runs/scored_bond_v3_20261117/target_audit.json \
  --output "${BB2M_OUTPUT:-runs/scored_bond_v3_formal_20261117}" \
  --epochs 20 --batch-size 8 --workers 8 --freeze-base-epochs 1 "$@"
