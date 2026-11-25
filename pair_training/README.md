# 共享 encoder 的原子对连接分支：第一阶段

保留 epoch119 的 backbone、encoder、atom decoder 和 line decoder。新增约 90 万参数的 Pair head：共享三尺度 encoder 特征，经投影后沿原子对连接区域的三条平行路径采样，并加入端点特征、原子类别和相对位置。当前第一阶段使用 Pair head 独立预测，不修改 WebApp，不合并旧 line decoder 输出，也不加载新权重到线上。

## 标签与数据

`pipeline/prepare_pair_data.py` 读取 real/*.csv 中名称包含 SMILES 的列。使用 RDKit canonical isomeric SMILES 去重，去除 atom map 编号；按不区分立体信息的 canonical SMILES 分组划分 train/val。同一分子的立体异构体和绘图版本不会跨集合。排除现有 verified benchmark 的分子连接结构。

当前只接受 2–70 个原子、有键、单组分、现有 atom/charge 类别可表示的分子。无效 SMILES、同位素/自由基、未支持元素和键型等写入 rejected.json；不静默截断盐、不把配位键当作单键。原来的 real 图片不参与自动标注。

Indigo 重新绘图，导出逐原子真实高亮像素 box、中心、类别、电荷、键端点与原子 ID、键型和 Molfile。检查绘图前后 canonical isomeric SMILES 一致，并用 Molfile 布局独立审计高亮中心；失败的图片剔除，不使用仿射估算 box。少于四个原子的图缺乏布局审计锚点，仍要求像素高亮成功、图结构一致。

Pair 类别：无键、other、single、double、triple、aromatic、正向 wedge/dash、反向 wedge/dash，共 10 类。`other` 不等于无键。KNN 使用每个原子最近 6 个邻居的无向并集，不包含自身；每对只能预测一种类别。普通键无向，楔形键方向单独编码。

## 第一阶段训练

旧网络全部冻结，且保持 eval 模式，防止 BatchNorm 缓冲区更新。训练只对 Pair head 反向传播。每张图片以 50% 概率用预测原子，否则用带少量坐标扰动的 GT 原子；训练时随机重排 GT 原子顺序，避免楔形方向依赖原始 atom ID。预测原子通过带距离门限的 Hungarian 匹配产生训练标签；无法匹配的原子对忽略，不错误标成无键。所有可监督近邻无键对参与分类损失。

验证分别报告 GT atoms 和 predicted atoms，推理分支不读取 GT。FN 分母包含不在 KNN 候选中的真实键。typed TP 同时要求键型和 wedge/dash 方向正确。exact_atom_bond_rate 不含电荷、同位素或最终 SMILES 的检查；不能当成 WebApp 最终化学结构准确率。预测原子来自原始 atom head + NMS，不含 WebApp 的 OCR 修正。

## 执行命令

环境已使用 `/home/user/miniconda3/envs/bl2m/bin/python` 验证。先生成正式图片与标签，再启动正式训练：

```bash
cd /data/bo/works/pys/portable_mp123_epoch119_20261013
bash scripts/prepare_pair_stage1.sh > runs/pair_real_prepare.log 2>&1
CUDA_VISIBLE_DEVICES=2 bash scripts/train_pair_stage1.sh > runs/pair_stage1_real.log 2>&1
```

默认数据目录 `dataset_pair_real_20261015`；完整语料每个分子画 3 个版本，上限为分子数 × 3，实际数量取决于渲染审计。正式图片尚需上述准备命令生成。该命令可以用相同参数重跑恢复；输入 CSV 内容变化需要新 corpus 目录。仅第一阶段正式训练交给用户执行，smoke 权重不能用于判断模型质量。

默认单卡 GPU 2、batch 4、30 epochs、lr 2e-4。用 `CUDA_VISIBLE_DEVICES` 指定物理 GPU 后，程序内使用 cuda:0。训练保存独立的新头权重、optimizer、随机数状态及数据/基模型指纹：

- `runs/pair_stage1_real_20261015/last.pth`：恢复训练。
- `best.pth`：按验证集中 predicted atoms 的 typed-bond F1 选择，同时扫描 0.3/0.5/0.7/0.9 阈值。
- `history.json`、`summary.json`：训练曲线、误键/漏键/原子指标、冻结状态和内存占用。

这是验证集选出的阈值与 checkpoint，最终收益仍应在保留基准集上检查。旧 epoch119 训练分子的完整清单未知，无法证明新验证集与旧 checkpoint 的训练语料完全独立。

恢复（EPOCHS 表示总 epoch 数，其他训练参数须与之前相同；恢复后保留 checkpoint 中 optimizer 的学习率）：

2026-10-16 修复：早期版本把单个 batch 的零梯度视为异常，会在模型预测饱和后中断训练。现在正常跳过该 batch 的 optimizer 更新并记录 `zero_gradient_batches`；非有限梯度和损失与可训练参数断开仍会报错。现有 checkpoint 可直接恢复，无需重新生成数据。当前正式运行完整保存到第 14 轮，best 为第 8 轮。

```bash
CUDA_VISIBLE_DEVICES=2 RESUME_CHECKPOINT="$PWD/runs/pair_stage1_real_20261015/last.pth" \
  bash scripts/train_pair_stage1.sh >> runs/pair_stage1_real.log 2>&1
```

可用环境变量调整 `EPOCHS`、`BATCH_SIZE`、`PAIR_LR`、`OUTPUT_DIR`、`DATASET_DIR`。先试较小正式集合时可设置 `MOLECULE_LIMIT=5000 DATASET_DIR=... bash scripts/prepare_pair_stage1.sh`，随后训练使用相同 DATASET_DIR；改变数据生成参数需换目录。

## 人工 Fix 的联合训练：独立 bond decoder

`train_human_joint.py` 会直接监督现有 `EndpointLineTransformer`：从共享图像特征预测 bond line 两端坐标与 bond 类别，训练标签来自 Fix JSON 的 `bonds[].line` 和 `category_id`。这条支路不先用预测 atom 生成 KNN 候选；推理组装时再按线端点与 atom 中心的几何关系确定连接对象。训练同时保留 pair head，便于与独立 line 结果做对照或后续混合；line decoder、backbone、encoder、atom/charge decoder 都参与更新。

用最新一次全权重 checkpoint 继续时，`best_base.pth` 与同目录的 `best_pair.pth` 必须配对。验证会单独报告独立 line decoder 的 bond 类型/端点 F1 和整图线段完全匹配率。单批 smoke 只检查训练链路，不作为性能结论；正式结果需要在全部 fresh holdout 上比较独立 line、pair 和混合输出。

## 独立图片推理

```bash
CUDA_VISIBLE_DEVICES=2 /home/user/miniconda3/envs/bl2m/bin/python -m pair_training.predict \
  --checkpoint runs/pair_stage1_real_20261015/best.pth \
  --image /absolute/path/to/image.png --output runs/pair_prediction.json
```

输出实验分支的原子、键、置信度 JSON，以及粗蓝色半透明线叠图。不运行 OCR、化学校验或与原有 bond head 融合。先确认漏键改善且误键受控，再开展联合训练与生产接入。

训练后在保留的 verified benchmark 上独立评估：

```bash
CUDA_VISIBLE_DEVICES=2 /home/user/miniconda3/envs/bl2m/bin/python -m pair_training.evaluate \
  --checkpoint runs/pair_stage1_real_20261015/best.pth \
  --dataset dataset_benchmark_verified_audit_20261014 \
  --output runs/pair_stage1_real_benchmark.json
```

报告同时列出多个阈值，正式比较优先使用 checkpoint 中已选定的验证阈值；不要在该基准上再次挑阈值后把结果当成独立测试成绩。
