#!/usr/bin/env bash
# Reproducible optional PaddleOCR comparison on frozen predicted graphs.
# Usage: bash scripts/evaluate_paddle_graph_ocr.sh OUTPUT_DIR [CACHE] [EASYOCR_DIR] [REFERENCE_SAMPLES] [DATASET]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
RUN_DIR="${1:?Specify an output directory}"
CACHE="${2:-runs/scored_bond_improvements_20261119/selected/prediction_cache/predictions.jsonl.gz}"
EASY_DIR="${3:-runs/scored_bond_improvements_20261119/selected/graph_ocr_verified}"
REFERENCES="${4:-runs/scored_bond_v3_comparison_20261118/reference_verified/samples.json}"
DATASET="${5:-/mnt/hdd2/bo/work/cvs/chemcv_data/real/indigo_full_v3_20261116}"
MAIN_PY="${PYTHON_BIN:-python}"
OCR_PY=.venvs/paddleocr311/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=2
"$MAIN_PY" -m band_training.prepare_paddle_ocr_requests --cache "$CACHE" --easyocr "$EASY_DIR" --dataset "$DATASET" --output "$RUN_DIR/requests.json"
if [[ ! -f "$RUN_DIR/recognition/candidates.json" ]]; then
  "$OCR_PY" band_training/paddle_multiscale_worker.py --requests "$RUN_DIR/requests.json" --output "$RUN_DIR/recognition" --models webapp_data/paddleocr_models
fi
"$MAIN_PY" -m band_training.compare_paddle_ocr --root "$RUN_DIR" --cache "$CACHE" --easyocr "$EASY_DIR" --references "$REFERENCES" --dataset "$DATASET"
