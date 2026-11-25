# 内部字符序列头：实验阶段

当前优先评估直接复用预训练识别网络，避免从零学习字符。已实现冻结 EasyOCR
PyTorch 子模块并接入 notebook 可选后端，见 [PRETRAINED_OCR.md](PRETRAINED_OCR.md)。
下面的 62 字符辅助预训练路线保留为备选，不是采用预训练 OCR 的前置条件。

后续主路线为 **62 字符自由序列识别**：a-z、A-Z、0-9，加一个 CTC blank，
输出层共 63 类。新模型默认字符表已限制到该范围；旧 checkpoint 按其保存的
`config.characters` 恢复，不会因默认值修改而改变旧模型类别顺序。
缩写仍是一个实体节点，框内字符不是额外的分子原子。缩写词表解码只保留作
历史对照，不作为新模型的默认解码，也不强制替换未见过的字符串。

`scripts/build_character_pretraining.py` 生成辅助字符预训练数据：全部 62 字符
都有正例，包含单字符、多字符、大小写、数字及相邻重复字符；不同 split 的
整串文本不重复，检验已见字符的新组合。数据来自四种本地字体、多个字号和
颜色，字体风格并非完全隔离。它是文字辅助数据，没有虚构化学图或展开 SMILES。
真正分子图的 CXSMILES/SMILES 对比仍使用独立的化学真值集。

小规模数据准备与训练入口：

```bash
/home/user/miniconda3/envs/bl2m/bin/python scripts/build_character_pretraining.py \
  --out dataset_character_pretraining --train-per-character 8 --eval-per-character 2
/home/user/miniconda3/envs/bl2m/bin/python scripts/train_text_sequence.py \
  --manifest dataset_character_pretraining/labels.jsonl --architecture native \
  --require-full-alphabet --require-unseen-strings \
  --steps 5000 --out runs/character62_pretrain
```

这只是小规模辅助字符预训练入口，不是最终整词检测/成键联合训练命令。
需要随后混入真实分子背景、尺寸和上下标样式，并继续用独立分子测试组合泛化。
CTC 不要求手工给每一个字符画框，整词框加完整字符串即可；连续重复字母通过
blank 分隔学习。当前固定 48 个时间步仍限制最长输出，不意味着支持无限长文本。
字符覆盖检查会在加载基础模型前拒绝缺失字符或非法字符的训练集。

目标是正确的连接图、完整标签节点和原样字符串。缩写展开可选；无法展开的
文字仍附着到对应 `*`，写入 CXSMILES。专项诊断确认长文字中心会影响连键，
现已在 NativeROITextHead 中加入锚点预测分支；锚点坐标本身不是最终验收指标。
新专项数据和对比结论见 [TEXT_BENCHMARK_REPORT.md](TEXT_BENCHMARK_REPORT.md)。

`text_sequence.py` 实现共享 encoder 三层特征 → 完整标签 ROI 采样 → 小型卷积
→ 双向 GRU → CTC 字符序列。不调用 PaddleOCR 或其他外部 OCR；大小写保留，
不做字典替换。`recognize_shared_labels` 返回可直接传给
`notebook_ocr.labeled_cxsmiles` 的节点文字映射，该序列化函数不启动 OCR。
置信度是未经校准的解码路径分数，不应直接使用原有 OCR 的阈值。

`NativeROITextHead` 另加原图高分辨率文字 CNN / GRU，与共享特征字符分支融合。
锚点分支复用文字 CNN，并采样文字周围的键线，输出与框分离的连接位置。
`decode_ctc_lexicon` 可按训练词表的 CTC 序列似然选择完整词，但它会强制把
陌生文字改成已知词；条件词表置信度不能当作“这是已知词”的置信度。

当前分支只负责读字，输入必须是完整标签框。它不会自动修好原 atom decoder
把 OMe 分裂成 O / e 的问题。完整替换外部 OCR 还需训练现有检测 decoder：
一个完整标签对应一个 `*` GT，字符碎片不作为独立目标。之后联合评估完整框、
字符串、节点连接和 CXSMILES；不能用真值框读字的准确率替代端到端结果。

## Smoke testing

```bash
/home/user/miniconda3/envs/bl2m/bin/python -m unittest discover -s tests -p test_text_sequence.py -v
/home/user/miniconda3/envs/bl2m/bin/python scripts/train_text_sequence.py \
  --smoke --steps 1000 --out runs/text_sequence_smoke_1000_20261019
```

Smoke 仅 32 个训练图、16 个独立渲染验证图，四种字符串 OMe / Boc / NH2 / R1，
使用 PIL 绘制文字、颜色和干扰线。训练与验证共享词表和字体，使用真值完整框。
这检验训练链路、冻结 encoder、权重重载，不代表新分子或实际化学绘图的识别率。
不加载或替换 WebApp 的模型，也不更改已有 pair 权重。

2026-10-19 实测：1000 步后训练整串正确 32/32，独立渲染验证 12/16；
冻结基础模型哈希不变，保存后重载输出完全一致。3 个新单元测试及 8 个既有
文字/CXSMILES 回归测试通过。300 步的较短实验只有 4/16，不应据此上线。
进一步在 notebook 的 medium_ocr_512 / large_ocr_768 上，用真值框测试 OMe/Boc，
**0/4 正确**。该结果明确表明这份小规模 PIL 字体权重不能替代当前外部 OCR。
需要真实分子渲染、多尺度/字体监督后再判断效果，不能将失败归为框检测问题，
因为此项测试已经提供真值框；也不能用这四个样本断言架构上限。
记录见 `runs/text_sequence_smoke_1000_20261019/summary.json` 和
`notebook_cases_gt_boxes.json`。

## 小规模标注数据实验

输入 JSONL，每行示例：

```json
{"image":"images/a.png","split":"train","group":"canonical_molecule_id","labels":[{"box":[100,120,165,145],"text":"OMe"}]}
```

box 为原图像素坐标 xyxy，必须包含整个标签；text 来自渲染真值，不能是由当前
OCR 生成的伪真值。train / val 的 molecule group 不能重叠。空文字标签可用作
背景负样本。支持字符由 checkpoint 的 config.characters 保存；不支持的字符
直接报错，不会悄悄删掉。CTC 时间长度不足也会报错。

```bash
/home/user/miniconda3/envs/bl2m/bin/python scripts/train_text_sequence.py \
  --manifest /absolute/path/to/text_labels.jsonl \
  --steps 1000 --out runs/text_sequence_pilot
```

以上为小规模字符头实验命令，尚不是全量联合训练方案。当前脚本将冻结特征
缓存到 CPU 内存，640 输入约每图 8.6 MB，不能直接用于数万张全量训练。
下一阶段需要流式训练、多字体及尺寸、真实分子背景、未见缩写与骨架划分，
同时修复原渲染标签提取中可能截断长文本框的问题。
