#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_ROOT="${BB2M_EVAL_OUTPUT:-runs/graph_metrics_$(date +%Y%m%d_%H%M%S)}"
DATASET_ROOT=/mnt/hdd2/bo/work/cvs/chemcv_data/real/indigo_full_v3_20261116
export CUDA_VISIBLE_DEVICES="${BB2M_GPU:-0}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$PYTHON_BIN" -m band_training.cache_final_predictions --checkpoint "${BB2M_CHECKPOINT:-runs/full_v3_paired_formal_20261117/best_joint.pt}" --output "$EVAL_ROOT/prediction_cache"
"$PYTHON_BIN" -m band_training.audit_graph_references --dataset "$DATASET_ROOT" --output "$EVAL_ROOT/reference_verified"
"$PYTHON_BIN" -m band_training.evaluate_graph_ocr --dataset "$DATASET_ROOT" --cache "$EVAL_ROOT/prediction_cache/predictions.jsonl.gz" --ocr-models webapp_data/ocr_models --output "$EVAL_ROOT/graph_ocr_verified"
"$PYTHON_BIN" -m band_training.rescore_functional_groups --evaluation "$EVAL_ROOT/graph_ocr_verified" --references "$EVAL_ROOT/reference_verified/samples.json"
"$PYTHON_BIN" -m band_training.finalize_graph_metrics --root "$EVAL_ROOT"
