#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m unittest \
  band_training.test_scored_bonds \
  band_training.test_line_neighborhood \
  band_training.test_paddle_ocr_replay \
  band_training.test_molecular_graph_metrics \
  band_training.test_v6_continue \
  band_training.test_v6_prior_control \
  band_training.test_invariants \
  band_training.test_dataset_runtime
"$PYTHON_BIN" -m band_training.predict_scored_bonds --help >/dev/null
"$PYTHON_BIN" -m band_training.train_v6_continue --help >/dev/null
