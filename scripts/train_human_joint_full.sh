#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/user/miniconda3/envs/bl2m/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
DEVICE="${DEVICE:-cuda:0}"
DATASET="${DATASET:-$PROJECT_ROOT/runs/pair_human_only_all_20261026}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$PROJECT_ROOT/weights/checkpoint0119.pth}"
INIT_PAIR="${INIT_PAIR:-$PROJECT_ROOT/runs/pair_robust_full_20261018/best.pth}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/runs/human_joint_full_$(date -u +%Y%m%d_%H%M%S)}"

[[ -f "$DATASET/manifest.json" ]] || { echo "Dataset missing: $DATASET" >&2; exit 1; }
[[ -f "$INIT_PAIR" && -f "$BASE_CHECKPOINT" ]] || { echo "Initialization checkpoint missing" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || { echo "Choose a new OUTPUT directory: $OUTPUT" >&2; exit 1; }

# Fixed epoch count, no early stopping. Only human Fix/augmentation images are
# sampled for gradients; the independent fresh validation images are evaluation only.
exec "$PYTHON_BIN" -m pair_training.train_human_joint \
  --dataset "$DATASET" \
  --init-pair "$INIT_PAIR" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --output "$OUTPUT" \
  --epochs "${EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --workers "${LOADER_WORKERS:-0}" \
  --lr-backbone "${LR_BACKBONE:-0.000001}" \
  --lr-encoder "${LR_ENCODER:-0.000002}" \
  --lr-detector "${LR_DETECTOR:-0.00001}" \
  --lr-line "${LR_LINE:-0.00001}" \
  --lr-pair "${LR_PAIR:-0.00001}" \
  --pair-weight 1.0 \
  --detection-weight 1.0 \
  --predicted-fraction 0.5 \
  --lowres-aug 0.25 \
  --k 6 \
  --score-mode existence \
  --device "$DEVICE"
