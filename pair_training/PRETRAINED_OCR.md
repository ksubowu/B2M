# 直接复用预训练 OCR

已完成零训练集成和评估：EasyOCR 权重没有更新，4 个裁剪的文字及置信度与
官方 API 一致。在同一真值整框、同一四尺度协议下，测试 108 个标签中
EasyOCR 为 26/108（24.1%），现有 PaddleOCR 为 54/108（50.0%）。这不是
端到端整图率。未适配的 EasyOCR 不应替换默认后端；更合理的是复用预训练网络，
针对化学小字、字号、字形和上下标做必要的领域微调。

Notebook 已增加预训练/适配识别子模块 `model.text_reader`。当前默认选择已更新，
见 [STRUCTURE_PROFILE.md](STRUCTURE_PROFILE.md)；`INFERENCE_PROFILE='legacy_paddle'` 可恢复原流程。整词区域检测继续沿用现有 Paddle
流程，因此这个选项不代表整个处理流程都已移除 Paddle。两条路径都不需要
从零训练字符，但仍要验证文字框和节点连接是否正确。

可以先省掉从零训练字符网络的过程。当前实现位于 `pretrained_ocr.py`：
将 EasyOCR 英文 english_g2 的完整 PyTorch 视觉编码器、双向 LSTM 和 CTC
分类器封装成冻结子模块。它不需要化学缩写词表，能直接输出大小写敏感的字符
组合；解码限制为 a-z / A-Z / 0-9，保持预训练输出层和原始类别顺序不变。

这里保留 OCR 自己的文字图像 encoder，并没有强行让它的 decoder 接收现有
化学 encoder 的特征。若只移植 decoder、要求两个任务共享同一 encoder，
特征分布和维度不匹配，通常仍需适配训练或蒸馏，不能称为零训练融合。

本模块的 forward 是实际的 PyTorch 模型推理；`recognize_regions` 包含原图
裁剪及 EasyOCR 标准预处理，无需启动外部进程或远程 OCR 服务。它要求调用方
提供完整文字区域，不会自行检测整词框，也不会把字符拆成多个分子原子。
锚点分支与文字识别是独立问题。

```python
from pair_training.pretrained_ocr import PretrainedEasyOCR, labels_from_regions
from pair_training.notebook_ocr import labeled_cxsmiles

ocr = PretrainedEasyOCR(model_dir=PROJECT_ROOT / 'weights/easyocr', device='cpu',
    checkpoint=PROJECT_ROOT / 'weights/text_profiles/chemical_adapted_v1/ocr.pth')
predictions = ocr.recognize_regions(image, whole_label_boxes_xyxy)
labels = labels_from_regions(predictions, graph_atom_indices)
labeled_mol, cx_result = labeled_cxsmiles(mol, graph, labels)
```

返回文字原样保存，缩写能否展开不影响 CXSMILES 输出。不同 OCR 模型的分数
不能直接比较。当前四尺度只在同一个 EasyOCR 模型内部选择最高置信结果。

## 数据与验证

复用 `dataset_text_benchmark_audited_20261019` 的既有划分，不改变测试真值。
`dataset_pretrained_ocr_crops_20261019` 已导出适合后续验证/少量领域微调的
文字裁剪、原文 TSV，以及对应整图真值 CXSMILES/SMILES 的路径。训练集的
图像增强只是相同文字区域的视觉变体，不能作为新增独立分子或新字符串计数。
其中训练 954 张（318 个真实文字区域及其增强），验证 104 张，测试 108 张，
源分子/骨架划分保持隔离。它仅用于识别验证和可选微调，不是完整 62 字符
覆盖的从零预训练集；陌生字符的初始能力来自 OCR 预训练权重。

应先测预训练模型，再根据失败样本决定是否微调。依然需要验证集，以及完整
标签定位、连接锚点和最终带文字分子图的监督；不必先投入大规模随机字符
从零训练。现有从零字符训练入口保留为备选，不是当前必走步骤。

```bash
/home/user/miniconda3/envs/bl2m/bin/python scripts/evaluate_pretrained_easyocr.py \
  --dataset dataset_text_benchmark_audited_20261019 \
  --out runs/text_recognizer_comparison_audited_20261019
```

可选依赖在 `requirements-easyocr.txt`；当前 bl2m 用 `--no-deps` 安装，保持原有
torch/torchvision/numpy/opencv 不变。模型位于 `weights/easyocr/english_g2.pth`，
初始化默认禁止下载；只在明确准备模型时允许联网。模型输出的识别质量仍需
实际化学图评估，不能因为加载了预训练权重就承诺完全不需要领域微调。

参考：[EasyOCR 官方 API](https://www.jaided.ai/easyocr/documentation/)、
[官方识别网络源码](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/model/vgg_model.py)。
