# B2M — Bond Bands to Molecular Graphs

B2M 用于化学结构图片的模型训练和预测：检测 atom/charge boxes，分别预测带 score
的 bond lines 和 bands，再通过几何关系组装分子图。line 端点表达连接位置，band
区域与类别表达单键、双键、三键、芳香键、实楔键和虚楔键的视觉特征。

**训练数据的生成、绘图及标注准备在独立仓库
[chem-structure-prep](https://github.com/ksubowu/chem-structure-prep)。**
本仓库只消费已准备好的图片与标注，不包含数据集和模型权重。

## 1. 安装模型环境

```bash
git clone https://github.com/ksubowu/B2M.git
cd B2M
conda env create -f environment-bb2m.yml
conda activate bb2m
```

`environment-bb2m.yml` 是已有 Linux/CUDA 训练环境的版本快照：Python 3.12、
PyTorch 2.10 / torchvision 0.25、CUDA 12.8。其它平台需调整对应 PyTorch 构建。

如果需要训练时的完整分子图评估，还需安装独立化学工具包作为图校验后端，
无需运行其数据生成命令：

```bash
python -m pip install git+https://github.com/ksubowu/chem-structure-prep.git
conda create -n b2m-ocr python=3.11 pip
conda run -n b2m-ocr python -m pip install -r requirements_paddle_ocr.txt
```

设置 `B2M_PADDLE_PYTHON` 为 `b2m-ocr` 环境的 Python 路径，
`B2M_PADDLE_MODELS` 为本地 PaddleOCR 模型目录，其中应有
`en_source/inference.pdmodel` 和 `en_source/inference.pdiparams`。
默认路径分别为 `.venvs/paddleocr311/bin/python` 和
`webapp_data/paddleocr_models/`。模型文件需另行提供。
完整评估固定使用 PaddleOCR 英文多尺度＋投票；PaddleOCR 不与训练环境混装。

## 2. 训练模型

当前入口 `band_training.train_v6_continue` 是从兼容 joint 权重进行微调的流程，
不支持直接从随机初始化开始。初始化权重 schema 应为
`independent_scored_line_band_joint_v3`。

### 已准备数据的输入要求

- 数据集根目录包含固定的 `train.json`、`val.json` 和 `dataset_config.json`。
  当前 preflight 使用后者的 `validation_reference.manifest_sha256` 验证冻结验证集。
- 每条记录包含 `sample_id`、`group`、`image`、`annotation`、`band_target`、
  `line_target`，资产路径相对于数据集根目录。
- `annotation` 提供 atom/charge boxes 与参考连通关系；`band_target` 提供
  `labels`、`quads`；`line_target` 提供对应的 `labels`、`lines`。
- box、line、band 必须对齐同一张原图；loader 会对图片与坐标施加同一个
  640×640 letterbox 变换。训练集与验证集不可有相同样本或分组。

类别为 13 个 atom 类别、5 个 charge 类别、6 个 bond 类别。
Atom：`other, C, O, N, Cl, Br, S, F, B, I, P, H, Si`；charge：
`-4, -2, -1, +1, +2`（并非每类都有训练样本）；bond head 标签依次为
`single, double, triple, aromatic, wedge, dash`。

```bash
# 检查现成训练集的目标、坐标与划分，并保存本次训练清单
python -m band_training.preflight_v6 \
  --dataset /path/to/dataset --output runs/preflight --workers 4
# 如有已归档极小图，可额外传入：
# --mini-hard-manifest /path/to/mini_hard/data_manifest.json

# 训练 3 轮，关闭早停
CUDA_VISIBLE_DEVICES=0 python -m band_training.train_v6_continue \
  --dataset /path/to/dataset --preflight runs/preflight \
  --init /path/to/best_joint.pt --output runs/experiment \
  --epochs 3 --batch-size 16 --workers 4 --patience 0
```

可用参数：

- `--freeze-feature-steps`：前多少次更新冻结 backbone/encoder；
  设为大于总更新次数即可全程冻结。
- `--freeze-boxes`：冻结 atom/charge box 分支，只训练 bond heads。
- `--box-lr`、`--shared-bond-lr`、`--line-lr`、`--band-lr`：各分支学习率。
- `--training-manifest`：使用通过预检的训练样本子集。
- `--checkpoint-every`：恢复点保存间隔；`--resume`：恢复训练。
- `--patience 0`：不早停；完整参数见 `--help`。

输出包括 `best_joint.pt`、`last_joint.pt`、各轮权重和恢复点。
每轮在固定验证集记录 atom/charge/bond 指标、严格全图正确率、
分子图正确* 和分子图正确_ocr；`monitor/` 保存逐轮 CSV、指标曲线和 loss 曲线。

## 3. 模型推理与预测

单图神经网络预测不需要安装数据准备工具或 OCR 模型。
提供兼容 joint 权重和输入图片即可：

```bash
python -m band_training.predict_scored_bonds \
  --checkpoint /path/to/best_joint.pt \
  --image /path/to/structure.png \
  --output runs/demo/prediction --device cuda:0 \
  --line-threshold 0.25 --band-threshold 0.20 \
  --refine-line-peaks --stereo-from-band
```

使用 `--device cpu` 可以改为 CPU 推理。该命令写出：

- `prediction.json`：原图像素坐标下的 atom/charge boxes、lines、bands、
  键类型、连接关系、各分支 score 及未关联候选。
- `prediction.png`：原图与各预测分支的预览图。

组图使用预测坐标关联，不使用真实 atom index。该命令不会识别官能团文字，
也不输出经过 OCR 展开的最终 SMILES。Joint checkpoint 已包含 base tensors，
不需要其历史 `config.base` 路径指向的父权重文件。仅加载可信来源的权重。

## 4. 代码与测试

`band_training/` 包含当前模型、loss、训练、推理、组图与评估；
`config/` 是模型配置；`repo/engine/` 是检测引擎。
`pair_training/`、WebApp 与历史 launcher 保留用于兼容旧模型，
部分历史脚本有旧机器路径；新任务使用上面的明确 CLI。

```bash
bash scripts/check_source.sh
```

完整测试同样需要第 1 节的化学图评估依赖。

训练数据和模型权重不随仓库发布；历史聚合指标仅存放于 `docs/benchmarks/`。
原创代码采用 [MIT License](LICENSE)，第三方引擎等保留原许可证，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
