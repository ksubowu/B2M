# B2M — Bond Bands to Molecular Graphs

B2M is a research implementation for chemical structure image recognition. It
combines atom/charge **boxes** with independently scored bond **lines** and
**bands**, then assembles a molecular graph using predicted geometry.

B2M 将化学结构图片识别为分子图：atom/charge boxes 定位原子和电荷，line
端点表达连接位置，band 区域及类别表达化学键的绘图特征。两种 bond head
分别输出 score；组图通过坐标关联完成，不需要模型输出 atom index。

## Architecture

- 13 atom categories: `other, C, O, N, Cl, Br, S, F, B, I, P, H, Si`.
- 5 charge categories: `-4, -2, -1, +1, +2` (not all have training examples).
- 6 bond categories: `single, double, triple, aromatic, wedge, dash`.
- Independent line and band heads share spatial features and instance queries.
  Lines predict ordered endpoints and a presence score. Bands predict a region,
  a presence score, and bond type.
- Postprocessing links line endpoints to predicted atom boxes, groups duplicate
  hypotheses by atom pair, selects band evidence, handles wedge/dash direction,
  and associates charge boxes. Ground-truth atom IDs are not inference inputs.
- Optional molecular label evaluation uses **PaddleOCR English PP-OCRv4**, crop
  scales 1/2/3/4 and margins 1/3, followed by normalized-text voting.

## Included source

| Path | Purpose |
| --- | --- |
| `band_training/` | Current box/line/band training, graph assembly, evaluation, diagnostics and tests |
| `chem_structure_prep/` | Portable Indigo rendering and box/line/band annotation preparation |
| `data_preparation/v6/` | Historical v6 dataset generation and validation scripts |
| `pair_training/` | Earlier pair-model experiments and shared utilities |
| `repo/engine/` | Bundled detection engine; retains Apache-2.0 licensing |
| `repo/i2bl_webapp/` | Historical manual annotation/review WebApp and Ketcher assets |
| `config/` | Model and training configurations |
| `scripts/` | Launchers; `scripts/experiments/` preserves historical machine-specific launch examples |
| `docs/benchmarks/` | Aggregate historical metrics and plots, without molecular sample data |

Training data, manually reviewed annotations, checkpoints, OCR model weights,
local environments, and run directories are **not** uploaded. Renderer toy
examples and required native renderer/font assets are included with notices.

## Environment

The working training environment was Python 3.12 with PyTorch 2.10 / torchvision
0.25, CUDA 12.8, NumPy 1.26.4, and RDKit. A pinned snapshot is provided:

```bash
conda env create -f environment-bb2m.yml
conda activate bb2m
python -m pip install -e ./chem_structure_prep
bash scripts/check_source.sh
```

`environment-bb2m.yml` is the tested machine's environment snapshot, not a
cross-platform lockfile. The bundled modified Indigo backend supports Linux
x86_64. The core detection pipeline requires separately supplied model weights.

Keep PaddleOCR 2.x in its separate Python 3.11 environment:

```bash
conda create -n b2m-ocr python=3.11 pip
conda run -n b2m-ocr python -m pip install -r requirements_paddle_ocr.txt
```

For training-time OCR validation, set `B2M_PADDLE_PYTHON` to that environment's
Python executable and `B2M_PADDLE_MODELS` to a directory containing
`en_source/inference.pdmodel` and `en_source/inference.pdiparams`. Alternatively,
use `.venvs/paddleocr311/bin/python` and `webapp_data/paddleocr_models/`.
The OCR worker requires local recognition model files; they are not downloaded
or included by this repository.

## Prepare data

```bash
chem-prepare doctor
chem-prepare init /tmp/b2m-example
chem-prepare run --config /tmp/b2m-example/recipe.yaml
# Existing unsplit datasets can receive paired bond line targets:
chem-prepare add-lines --dataset /path/to/unsplit --output /path/to/with-lines
```

See [the renderer documentation](chem_structure_prep/README.md). The training
manifest references images, atom/charge annotations, `band_target` and
`line_target`. Coordinates must share the same image frame; the training loader
applies the same 640×640 letterbox transform to pixels, boxes, lines, and bands.
Do not pass a dataset without line targets to the scored joint trainer.

## Predict an image

Provide a compatible `independent_scored_line_band_joint_v3` checkpoint:

```bash
python -m band_training.predict_scored_bonds \
  --checkpoint /path/to/best_joint.pt \
  --image /path/to/structure.png \
  --output runs/demo/prediction --device cuda:0 \
  --line-threshold 0.25 --band-threshold 0.20 \
  --refine-line-peaks --stereo-from-band
```

The command writes a graph JSON in original-image pixel coordinates and a
preview PNG. This CLI performs geometry-only graph prediction; it does not run
OCR. Joint checkpoints contain the base tensors, so their historical
`config.base` file path is not required by this publication's current loader.
Use only checkpoints from a trusted source: loading uses PyTorch pickle.

## Train and evaluate

The current continuation entry point is `band_training.train_v6_continue`.
It expects a compatible joint initialization checkpoint, a passed preflight,
fixed `train.json` / `val.json`, renderer-compatible annotations, and a local
PaddleOCR runtime. It does not train an initialized model from scratch.

```bash
python -m band_training.preflight_v6 \
  --dataset /path/to/dataset --output runs/preflight --workers 4
# Supply --mini-hard-manifest /path/to/mini_hard/data_manifest.json when needed.

CUDA_VISIBLE_DEVICES=0 python -m band_training.train_v6_continue \
  --dataset /path/to/dataset --preflight runs/preflight \
  --init /path/to/best_joint.pt --output runs/experiment \
  --epochs 3 --batch-size 16 --workers 4 --patience 0
```

`--patience 0` disables early stopping. Other options include
`--training-manifest`, `--freeze-boxes`, `--freeze-feature-steps`, per-head
learning rates, and resumable checkpointing. Each epoch evaluates all fixed
validation images with identical graph rules and PaddleOCR settings. Monitors
are written under the run's `monitor/` directory. Historical launch scripts may
refer to unavailable local data or old checkpoints; use the explicit CLI above
for a new machine.

## Development evaluation record

The historical v6 dataset had **140,490 effective training images** (six known
mini_hard images excluded) and **1,497 fixed validation images**. The retained
pre-v6 best checkpoint achieved:

| Metric | Fixed validation |
| --- | ---: |
| Atom F1 | 99.65% |
| Charge F1 | 90.91% |
| Bond F1 | 99.09% |
| Strict complete graph | 92.25% |
| Molecular graph* | 92.12% |
| Molecular graph_ocr | 91.12% |

These are **development validation results**, not independent test estimates.
The reported atom F1 uses the historical label-aware matching rule (IoU ≥0.1
or a center-distance fallback); IoU-0.5 detection metrics are recorded
separately. Bond correctness includes both linked atoms, type, and directed
stereobond orientation. Graph* anonymizes functional-group/dummy labels;
graph_ocr additionally checks recognized text. Isotope, atom CIP, implicit H,
and E/Z are not independently scored by these metrics.

Longer v6 continuation and subsequent short controlled experiments did not
surpass the retained checkpoint on whole-molecule accuracy. Their plots and
aggregate CSVs are preserved in `docs/benchmarks/`; newer weights should not be
assumed better merely because they trained longer. No checkpoint download is
provided in this source snapshot.

## License

Original B2M code is released under the [MIT License](LICENSE). Bundled upstream
components keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and `repo/LICENSE`. This is a source-code publication, not a grant of rights to
external molecular datasets or model weights.
