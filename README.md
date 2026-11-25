# B2M — Bond Bands to Molecular Graphs

B2M trains models to recognize chemical structure images and predict molecular
graphs. It detects atom/charge boxes, predicts independently scored bond lines
and bands, and assembles a graph using geometric relationships. Line endpoints
locate connections; band regions and classes capture the visual features of
single, double, triple, aromatic, solid-wedge, and dashed-wedge bonds.

**Image rendering, training-data generation, and annotation preparation are
maintained separately in
[chem-structure-prep](https://github.com/ksubowu/chem-structure-prep).**
B2M consumes prepared images and annotations. Datasets and model weights are
not included in this repository.

## 1. Install the model environment

```bash
git clone https://github.com/ksubowu/B2M.git
cd B2M
conda env create -f environment-bb2m.yml
conda activate bb2m
```

`environment-bb2m.yml` is a snapshot of the working Linux/CUDA training
environment: Python 3.12, PyTorch 2.10 / torchvision 0.25, and CUDA 12.8.
Other platforms require an appropriate PyTorch build.

Complete molecular graph evaluation during training also requires the separate
chemistry package as a graph-validation backend. You do not need to run its
data-generation commands:

```bash
python -m pip install git+https://github.com/ksubowu/chem-structure-prep.git
conda create -n b2m-ocr python=3.11 pip
conda run -n b2m-ocr python -m pip install -r requirements_paddle_ocr.txt
```

Set `B2M_PADDLE_PYTHON` to the Python executable in the `b2m-ocr` environment.
Set `B2M_PADDLE_MODELS` to a local PaddleOCR model directory containing
`en_source/inference.pdmodel` and `en_source/inference.pdiparams`.
The default paths are `.venvs/paddleocr311/bin/python` and
`webapp_data/paddleocr_models/`, respectively. Model files must be supplied
separately. Complete evaluation uses PaddleOCR English multiscale recognition
with voting. Keep PaddleOCR in its separate environment.

## 2. Train a model

The current entry point, `band_training.train_v6_continue`, fine-tunes a
compatible joint checkpoint. It does not support training directly from random
initialization. The initialization checkpoint must use the schema
`independent_scored_line_band_joint_v3`.

### Prepared dataset requirements

- The dataset root contains fixed `train.json`, `val.json`, and
  `dataset_config.json` files. The current preflight uses
  `validation_reference.manifest_sha256` in the configuration to verify the
  frozen validation set.
- Each record contains `sample_id`, `group`, `image`, `annotation`,
  `band_target`, and `line_target`. Asset paths are relative to the dataset root.
- `annotation` provides atom/charge boxes and reference connectivity;
  `band_target` provides `labels` and `quads`; `line_target` provides the
  corresponding `labels` and `lines`.
- Boxes, lines, and bands must align with the same original image. The loader
  applies the same 640×640 letterbox transform to the image and coordinates.
  Training and validation sets must not share samples or groups.

The model has 13 atom classes, 5 charge classes, and 6 bond classes:

- Atoms: `other, C, O, N, Cl, Br, S, F, B, I, P, H, Si`.
- Charges: `-4, -2, -1, +1, +2`. Not every class has training examples.
- Bond-head labels, in order: `single, double, triple, aromatic, wedge, dash`.

```bash
# Check targets, coordinates, and splits; save the effective run manifests.
python -m band_training.preflight_v6 \
  --dataset /path/to/dataset --output runs/preflight --workers 4
# To exclude previously archived extremely small images, optionally add:
# --mini-hard-manifest /path/to/mini_hard/data_manifest.json

# Train for three epochs with early stopping disabled.
CUDA_VISIBLE_DEVICES=0 python -m band_training.train_v6_continue \
  --dataset /path/to/dataset --preflight runs/preflight \
  --init /path/to/best_joint.pt --output runs/experiment \
  --epochs 3 --batch-size 16 --workers 4 --patience 0
```

Available options include:

- `--freeze-feature-steps`: freeze the backbone/encoder for the specified number
  of updates. Set it above the total update count to keep them frozen throughout.
- `--freeze-boxes`: freeze the atom/charge box branch and train only bond heads.
- `--box-lr`, `--shared-bond-lr`, `--line-lr`, `--band-lr`: learning rates for
  the respective branches.
- `--training-manifest`: use a subset of training records that passed preflight.
- `--checkpoint-every`: recovery checkpoint interval; `--resume`: resume training.
- `--patience 0`: disable early stopping. See `--help` for all options.

Outputs include `best_joint.pt`, `last_joint.pt`, per-epoch checkpoints, and
recovery checkpoints. Each epoch records atom/charge/bond metrics, strict
complete-graph accuracy, anonymous-dummy molecular graph accuracy (`graph*`),
and molecular graph accuracy with OCR (`graph_ocr`) on the fixed validation set.
The run's `monitor/` directory contains per-epoch CSV files, metric curves, and
loss curves.

## 3. Run inference and make predictions

Single-image neural prediction does not require the data-preparation package or
OCR models. Supply a compatible joint checkpoint and an input image:

```bash
python -m band_training.predict_scored_bonds \
  --checkpoint /path/to/best_joint.pt \
  --image /path/to/structure.png \
  --output runs/demo/prediction --device cuda:0 \
  --line-threshold 0.25 --band-threshold 0.20 \
  --refine-line-peaks --stereo-from-band
```

Use `--device cpu` for CPU inference. The command writes:

- `prediction.json`: atom/charge boxes, lines, bands, bond types, connectivity,
  branch scores, and unlinked candidates in original-image pixel coordinates.
- `prediction.png`: a preview of the original image and prediction branches.

Graph assembly associates predicted coordinates without ground-truth atom
indices. This command does not recognize functional-group text or produce a
final SMILES with OCR-expanded labels. Joint checkpoints already contain the
base tensors, so the parent checkpoint referenced by the historical
`config.base` path is not required. Load checkpoints only from trusted sources.

## 4. Source layout and tests

`band_training/` contains the current models, losses, training, inference,
graph assembly, and evaluation. `config/` contains model configurations, and
`repo/engine/` contains the detection engine. `pair_training/`, the WebApp,
and historical launchers are retained for compatibility with earlier models.
Some historical scripts contain machine-specific paths; use the explicit CLI
commands above for new tasks.

```bash
bash scripts/check_source.sh
```

The complete test suite also requires the chemistry evaluation dependency
listed in Section 1.

Training data and model weights are not distributed with this repository.
Historical aggregate metrics are retained under `docs/benchmarks/`.
Original code is licensed under the [MIT License](LICENSE). Third-party
components retain their original licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
