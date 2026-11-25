#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export BASE_CHECKPOINT="${BASE_CHECKPOINT:-$PROJECT_ROOT/runs/human_joint_full_20261027/best_base.pth}"
export INIT_PAIR="${INIT_PAIR:-$PROJECT_ROOT/runs/human_joint_full_20261027/best_pair.pth}"
export DATASET="${DATASET:-$PROJECT_ROOT/runs/pair_human_only_all_20261026}"
exec bash "$PROJECT_ROOT/scripts/train_human_joint_full.sh" "$@"
