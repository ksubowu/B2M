# 低清漏键与键型混淆改进，2026-10-18

已发布可独立推理的实验模型：`runs/pair_adaptive_20261018/model.pth`。旧 epoch119、第一阶段模型和 WebApp 默认权重均保留。

## 根因与实现

第一阶段第29轮模型在旧300张源数据中，77条键型错误有73条是单键错成 wedge/dash，76条键型错误来自低清S2黑白绘图。20条缺失连接中，16条的总成键概率接近1，但最大单类概率低于0.7；把键型置信度同时当作成键阈值，误删了本来存在的连接。

本次实现：

1. 在低清域分开监督、判断存在性和键型：存在性使用 `1-P(none)`，键型在非空类中选择，保存 wedge/dash 的方向。
2. 训练时先在160–320像素栅格上用面积采样、黑白阈值重采样，再送入640像素网络。归一化原子位置和拓扑标签不变。不是简单放大后又缩回原尺寸。
3. 比较原头增强微调、带原图定向局部CNN的新头、普通图片蒸馏三个小规模实验。单个微调头在清晰域存在退步。
4. 最终组合为两个独立键头共享原 backbone/encoder/atom decoder；推理每张图只执行相应的键头。原始栅格最短边<320时用低清头（当前总成键概率阈值0.7），其余用冻结的原清晰头及原0.7键型阈值。320是第一阶段原生训练图的最小栅格尺寸；路线根据栅格元数据确定，不读取GT/SMILES。

分辨率只是图像质量的代理。已经被其他软件放大的低清图片，或极端长宽比图片，不保证能被这个路由正确分类。这里验证的是当前绘图/上传流程；不是任何风格图片的通用零错误保证。原图CNN分支也保留在代码中，但本轮低清验证表现未超过选用的增强微调头。

## 测试范围

- 旧300张源数据已经用于错误诊断，属于开发基准。
- 新增100个分子，排除全部新头训练/验证语料与旧基准的分子连接结构，独立绘制3种样式。21次渲染/坐标/结构检查未通过，最终279张纳入评估。原始epoch119的训练清单未知，不能宣称整套模型完全未见过这些分子。
- 清晰验证集600张同时复测，确认保留原清晰头的所有原子/键判定。
- 阈值来自验证集，不在新测试集上重新搜索阈值。
- 指标为原子类别及键型图，不包含电荷、同位素、最终SMILES；也单独检查了楔形方向。

完整数字见 `runs/pair_robust_review_20261018/comparison.json`。案例原图、atom-box、蓝色半透明粗线叠图见 `runs/pair_adaptive_case_review_20261018/index.html`。

## 直接推理

```bash
cd /data/bo/works/pys/portable_mp123_epoch119_20261013
CUDA_VISIBLE_DEVICES=0 /home/user/miniconda3/envs/bl2m/bin/python -m pair_training.predict \
  --checkpoint runs/pair_adaptive_20261018/model.pth \
  --image /absolute/path/to/image.png --output runs/my_robust_prediction.json
```

输出JSON和半透明粗线PNG。JSON的 `profile` 标明实际使用的清晰/低清头。`score` 是该分支用于筛选的置信度，`type_probability_given_bond` 是条件键型概率。

## 进一步全量训练

当前低清头只在3,000张训练图片上做了2轮小规模对照，已超过原目标90%。剩余复杂含硫结构、少数键型与原子分类错误仍须继续处理，不能称为彻底解决。

已准备只读训练集的难例采样权重：磺酰基16,571张、四面体立体中心24,364张、大分子27,336张；采样分别为4倍/2倍/2倍，重叠时取最大值。没有把旧开发基准或新测试图放入训练。

按之前约定，全量训练仍由用户执行；本轮只执行了smoke与小规模对照。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_pair_robust.sh > runs/pair_robust_full.log 2>&1
```

默认完整训练集、5轮、batch4、lr2e-5。按低清验证集的整图正确率选择checkpoint。完成后自动把最佳低清头与冻结清晰头打包到 `runs/pair_robust_full_20261018/adaptive_model.pth`。这组完整训练设置已完成带难例采样的有限步数smoke，收益仍需要训练完成后重测，不能预先保证。

恢复时保持相同数据、采样文件和训练参数：

```bash
CUDA_VISIBLE_DEVICES=0 RESUME_CHECKPOINT="$PWD/runs/pair_robust_full_20261018/last.pth" \
  bash scripts/train_pair_robust.sh >> runs/pair_robust_full.log 2>&1
```

不要用第一阶段的last.pth恢复这个新实验；以第29轮基线初始化后开始新的训练阶段。

训练结束后复测：

```bash
CUDA_VISIBLE_DEVICES=0 /home/user/miniconda3/envs/bl2m/bin/python -m pair_training.evaluate \
  --checkpoint runs/pair_robust_full_20261018/adaptive_model.pth \
  --dataset dataset_benchmark_verified_audit_20261014 --output runs/pair_robust_full_benchmark.json
CUDA_VISIBLE_DEVICES=0 /home/user/miniconda3/envs/bl2m/bin/python -m pair_training.evaluate \
  --checkpoint runs/pair_robust_full_20261018/adaptive_model.pth \
  --dataset dataset_pair_fresh_holdout_20261018 --output runs/pair_robust_full_fresh.json
```
