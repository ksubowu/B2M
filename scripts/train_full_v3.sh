#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${BB2M_GPU:-0}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
exec "${PYTHON_BIN:-python}" -m band_training.train_full_v3 \
  --dataset /mnt/hdd2/bo/work/cvs/chemcv_data/real/indigo_full_v3_20261116 \
  --audit runs/full_v3_training_audit_20261116/report.json \
  --init-joint runs/indigo_charge_continue_20261114/inference_joint.pt \
  --output "${BB2M_OUTPUT:-runs/full_v3_paired_formal_20261116}" \
  --epochs 20 --batch-size 8 --workers 8 --freeze-feature-epochs 1 \
  --line-association exclusive_box_noisy --endpoint-order canonical "$@"
