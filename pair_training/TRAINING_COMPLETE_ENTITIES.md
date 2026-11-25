# OCR 适配、完整标签框和连接锚点训练

## 当前下一轮入口：正式训练关闭早停

按最新要求，`scripts/train_chemical_structure_dual_decoder.sh formal` 显式传 `--patience 0`，并在正式结束时强制检查**实际步数必须等于请求的3000、`stopped_early`必须为false**，另要求导出模型全量公开验证与训练内所选指标一致。每250步仍做整图评估和合格checkpoint选择，未通过分组保留的候选不会替换best，但训练本身继续跑满步数。入口自动创建UTC新RUN_DIR，本次新正式任务从已验证初始化重新开始，不覆盖之前的formal或pilot权重。

关闭早停后的新入口smoke已完成：8/8步、`patience=0`、`stopped_early=false`，模型真实更新、原检测/锚点头保持、保存重载一致，新旧域公开推理均完成。旧512px回退诊断指向**文字裁剪/识字敏感性**：例如 `T_val_001_512.png` 基线读出 `CF3`（0.992），学习分支在相近框位置读成 `C3`（0.998）；`T_val_055_512.png` 将 `tBu` 变为 `tBui`。图框、键数量相同，说明该例不是漏框造成。后续formal入口按原保留门槛运行完整3000步；小字OCR泛化不足的根因仍需针对性解决。

正式命令：

```bash
cd /data/bo/works/pys/portable_mp123_epoch119_20261013
DEVICE=cuda:1 bash scripts/train_chemical_structure_dual_decoder.sh formal \
  > "runs/dual_decoder_formal_$(date -u +%Y%m%d_%H%M%S).log" 2>&1
```

早停关闭smoke及正式完成条件记录：`runs/chemical_structure_dual_decoder_formal_audit_20261022/no_early_stop_preflight.json`。上节1250步并早停的结果是旧入口记录；本段才是之后运行的新入口要求。

## 2026-10-22 双decoder正式训练检查：正常早停，有改善但目标未达到

检查 `runs/chemical_structure_dual_decoder_formal_20261022_052153`：实际1250/上限3000步，最佳检查点250步；后续500/750/1000/1250四个候选均未通过旧512px及其已知缩写组保留，按patience=4正常早停。日志包含最终Finished，无Traceback/OOM/运行时异常；模型有真实更新，原检测/锚点头及冻结模块保持，BN统计保持。导出模型逐bit重载一致，独立全量公开验证与训练内选中模型逐分组完全一致。没有训练/验证进程残留，未新增test评估。

| 验证图 | 初始化v2 | 正式训练选中结果 |
| --- | --- | --- |
| 旧域589图 | 458/589，77.76% | 470/589，79.80% |
| 新域682图 | 220/682，32.26% | 239/682，35.04% |
| 旧256px | 121/190，63.68% | 132/190，69.47% |
| 新256px | 49/212，23.11% | 51/212，24.06% |
| 新512px | 78/235，33.19% | 89/235，37.87% |
| 新768px | 93/235，39.57% | 99/235，42.13% |

旧域修复23、回退11，净增12；新域修复37、回退18，净增19。各原验收分组正确数均保持，但不保证逐张不回退。选中模型仍只有新已知缩写99/140（70.71%）、未知字面串102/474（21.52%）；90%、小字可靠识读、彻底消除漏键及键型混淆均未实现。旧notebook默认v1此前为467/589，正式结果470仅比该默认多3张，表中初始化是v2的458，不能混淆比较基准。未切换默认。

候选曲线：250步旧470/新239；500步462/252；750步465/254；1000步466/259；1250步465/262。旧512px已知缩写组基线147/179，后四次分别145/146/145/146，所以即使最后新域更多图正确，也不能作为合格best。普通原子F1约98.15%→98.24%、文字框F1约93.84%→94.60%为选中模型局部指标；1250步文字框F1更高仍未转化为合格整图升级。这里证明的是配方触发保留早停，不是充分收敛；不建议原样延长。

与此前pilot公开成绩471/246相比，本次470/239略低。两次签名只在patience=0/4不同，数据、权重来源、学习率等相同；前25步loss已有差异，训练未要求严格确定性。GPU数值差异与框坐标/多尺度OCR选择敏感性各占多少尚未隔离验证，不能声称已精确复现pilot，也不能把pilot数字写成本轮正式成绩。

预测后几何匹配诊断（按图片计，类别非互斥、受匹配及芳香键表示影响）：新域缺原子212→192、漏键303→266、键型错误36→31、额外键90→94。漏键有所减少但远未解决，且有误键权衡。预测文字区域字面识别正确735→737；框覆盖GT文字≥95%仍读错405→412，缺文字区域82→68，说明完整框有所改善，读字仍是显著瓶颈。用GT文本替换几何匹配到的预测标签、保留预测结构的诊断，新域239→373/682，小图51→119/212；此为离线诊断，不是可部署成绩或严格上界。即便纠正这些文字，预测图错误仍明显。

建议先定位旧512px回退图的框/裁剪/读字/连接差异，再做不降低原分组要求的小字OCR及连键方案试验；后续训练仍应先证明整图净收益，再开始长训。原始错误图与测试集保持，不放宽验收或删除难图。

机器报告 `runs/chemical_structure_dual_decoder_formal_audit_20261022/final_diagnosis.json`，曲线 `training_curves.png`，文本替换诊断 `gt_text_substitution.json`。选中bundle位于 `runs/chemical_structure_dual_decoder_formal_20261022_052153/bundle`。以下为此前准备和试训记录，先前“未启动正式训练”和471/246均指当时状态。

## 2026-10-22 自主优化与验证记录

本轮使用 `dataset_text_retained_20261022` 和原生裁剪 `dataset_text_retained_crops_20261022`：训练3448图、691个独立源分子、614个源骨架、4358种字符串、6498个文字区域。新增840次绘图中831张通过独立化学/坐标检查，595张再通过原生可辨性与高分辨率参考检查，236张隔离。原训练图保留，旧隔离图未解禁。256px已知缩写区域从6增至720，512/768px为438/524；12种训练缩写在每个尺寸各至少53/30/37例且覆盖三种字体，TBDMS/NHCbz仍是留出词。每个尺度也完整覆盖62字符。原682验证和659测试图的像素、GT与顺序逐文件一致；测试未用于本轮选模。

清晰图复查：`dataset_text_retained_20261022/small_training_review.png`；236张隔离图浏览：`data_quarantine/text_readability_20261022/review.html`。原生小字只允许新的三倍矢量渲染采用7px墨迹/8px框高/3.5px平均字宽下限，并检查灰阶及参考墨迹质量；旧图继续使用原有严格规则。筛查旧已知缩写训练并排除当前留出骨架后，仅1张文字图通过原严格规则，因此本轮没有通过重新加入旧模糊/二值化文字来保留旧域。

OCR采用冻结原权重、可合并的序列投影/卷积残差和GT正确样本教师一致性，训练裁剪与推理采用相同缩放补白及按宽度分组，回放清晰已知词；真实周边键像素来自训练原图。视觉残差500步候选旧整图473/589（初始化458），旧小图133/190（原121），新整图221/682（原220），但新512px未知标签子组少1张，仍未通过原有全部分组保留条件。该候选标为未选中，不能把它写成本轮最终升级结果。

实体部分比较输出头残差、跨注意力残差、完整decoder方案。新增文字辅助损失的早期版本出现独立匹配/硬正例与原VFL质量分数冲突：文字召回98.94%，但预测框1579对GT1228，精度76.95%，不予采用；修正版统一匹配并移除硬正例。残差方案的完整验证仍未通过保留。完整单decoder对照采用冻结BN运行统计、按同一GT对象匹配教师与学生（允许proposal顺序变化），只保留小型native锚点残差，化学encoder/backbone/line decoder及pair权重不更新。250步新图达到237/682，但旧图部分控制/已知缩写分组退步，500步仍未通过，因此不推荐该单decoder入口正式长训。

当前方案改为共享encoder的双box decoder：冻结并随bundle保存原检测/锚点头，另训练完整文字实体decoder及小型锚点残差；两分支共用同一encoder特征和pair头。依据预测图本身选择：保留原分支有效且连通的无文字图；文字图优先学习分支，若它断裂或化学组装失败且原分支有效连通，则回退原分支。规则不读GT、不查缩写词表、不借助外部文字检测器。字母数字仍由领域适配的预训练OCR网络读取，保留原文CXSMILES。入口 `scripts/train_chemical_structure_dual_decoder.sh`。8步smoke、恢复到9步、公开权重重载及完整推理已验证，全套120项检查通过；notebook在两类实际选择分支上断言与CLI的原子/键完全一致，默认权重保持原样。故意空白OCR的回退实测证明未通过候选会恢复原OCR，不能只恢复检测器而遗留退步OCR。

250步双decoder pilot及独立公开bundle全量重载验证已完成，`selected_step=250`，两次成绩逐分组完全一致，全部原保留分组通过：

| 验证图 | 初始化v2 | 双decoder 250步 |
| --- | --- | --- |
| 旧域589图 | 458/589，77.76% | 471/589，79.97% |
| 新域682图 | 220/682，32.26% | 246/682，36.07% |
| 旧256px图 | 121/190，63.68% | 132/190，69.47% |
| 新256px图 | 49/212，23.11% | 54/212，25.47% |
| 新512px图 | 78/235，33.19% | 90/235，38.30% |
| 新768px图 | 93/235，39.57% | 102/235，43.40% |

旧域修复23张、回退10张，净增13；新域修复44张、回退18张，净增26。“分组不退”指各组正确图数，并非每一张旧正确图片都被保留。当前新域已知缩写99/140、未知字面串109/474，仍明显不足，90%以及彻底消除漏键/键型混淆均未达到。它是非零更新带来有效整图提升的短训证据，不是充分训练的结论。改进前后原图/原子框/半透明粗蓝键线见 `runs/chemical_structure_retained_audit_20261022/dual_decoder_cases.png`；按预先指定的域/尺寸/类型取首张新增正确图片，仅用于示例。

公开bundle：`runs/chemical_structure_dual_decoder_pilot_20261022/bundle`；原检测头223个tensor、原锚点43个tensor、其余base1126个tensor及pair24个tensor均与初始化逐项一致，学习decoder确有改变，OCR与已验证残差候选逐字节一致。notebook用该公开bundle验证四张val图片，覆盖保留/学习/回退三种选择及未知字面标签，图、SMILES和CXSMILES全部与CLI一致；原默认bundle未切换。

正式训练前预检已实际通过，但未启动正式长训。由用户执行：

```bash
cd /data/bo/works/pys/portable_mp123_epoch119_20261013
DEVICE=cuda:1 bash scripts/train_chemical_structure_dual_decoder.sh formal \
  > "runs/dual_decoder_formal_$(date -u +%Y%m%d_%H%M%S).log" 2>&1
```

入口使用bl2m的Python，自动生成新的UTC时间RUN_DIR，上限3000步，每250步检查完整旧/新val及全部原分组，连续4个验证间隔无选中改善则早停，保留最佳组合并独立重载全量验证。它从同一初始化重训以复现已通过的250步配方，不覆盖pilot，不直接续接被拒候选。正式训练后再检查充分性和独立泛化；本轮未再评估或调优test。

选模比较最终OCR/实体组合，而非要求每个中间分支独立改进；原v2整图与各尺寸、已知/未知标签、控制图交叉分组保留条件不变。所有GT仅用于训练或预测后的评分，不参与推理选框、连线、读字。正式入口按UTC生成新RUN_DIR，保留已完成检查点；正式前要求公开合并bundle全量验证与训练内验证一致、选到非零更新，并至少新增7张正确新验证图，smoke本身不能证明值得长训。

可独立使用的保守补键候选 `runs/chemical_structure_native_bridge_candidates_20261022/v2` 仅恢复不同连通分量之间、原图有同向连续线、键长与价态合理、不跨过第三个原子且键型线数一致的KNN边；保留现有键及原阈值，已有完整连通图立即返回。20条新增键在预测后独立与GT核对，20条连接/键型一致；原v2旧验证458→461/589，新验证220→222/682，各分组无回退。仅5张整图修复，因为其余图还有其它错误。CNN500加补键为476/589、223/682，但新中图子组仍退1张，保持未选中。这些比较使用初始化v2；notebook原v1旧验证467/589，不能据461就替换默认。

用GT字符串替换匹配区域、保留预测原子键的诊断为旧497/589、新336/682；这不是可部署成绩或严格上界，说明只修读字也无法达到90%。原始几何匹配诊断新图212张有缺原子、303张有缺键，统计非互斥且原始键类别受芳香表示影响。补键叠图 `runs/chemical_structure_retained_audit_20261022/native_bridge_cases.png` 使用半透明粗蓝线表示已有键，绿色表示新增键。单次正确20/20不构成普遍正确率保证。

完整动态报告 `runs/chemical_structure_retained_audit_20261022/final_report.json`，其中 `formal_ready`、双decoder公开验证结果与 `formal_command` 是最终执行依据；代码快照和SHA256在同目录，候选曲线 `candidate_curves.png`。正式长训未由助手启动，notebook默认权重未切换。开发中更新运行中的shell曾影响两次OCR pilot末尾导出；训练权重/总结完整保留，已补齐导出并以稳定入口重新完成smoke，见各run的 `completion_audit.json`。

## 2026-10-22 均衡小字正式训练检查：正常结束，但无有效升级

`runs/chemical_structure_balanced_stratified_formal_20261021` 三阶段都正常完成并触发早停：OCR实际3000/上限4000步，实体/锚点3000/6000步，pair3000/6000步；每阶段6个候选均未通过整图保留条件，最终selected_step均为0。四份最终权重base/anchor/pair/OCR逐tensor与初始化v2完全一致（1460个tensor），不是训练未更新，而是训练更新后的候选全部被拒绝。恢复/重载检查通过，末尾Finished及三套最终验证输出完整。

相同E/Z完整推理：旧589图458/589（77.76%），新682图220/682（32.26%），与初始化完全相同，提升0个百分点；新256/512/768分别49/212、78/235、93/235。GT框OCR仍为863/1228（70.28%），256/512/768分别161/378、327/426、375/424。最终模型没有实现小字改善、减少漏键/键型混淆或90%整图目标。本次没有新增独立测试评估，因为所有推理权重已证实完全相同；不能把被拒候选的局部指标当成最终成绩。

OCR最后候选GT框读字885/1228，但旧整图405/589（初始458），新整图224/682（初始220），新256图38/212（初始49）。实体最后候选文字框F1为97.22%（初始93.84%），新整图299/682（43.84%），旧整图437/589（74.19%）；仍牺牲旧域。pair各候选新域213–217/682、旧域429–444/589，都低于初始。拒绝不是仅由小分组严格限制导致：所有候选连旧域整体458/589都未保持。尺度均衡确实生效，三尺度各覆盖62字符，但不足以解决域间权衡。

不建议原样延长训练，也不把本bundle切为默认。下一轮先做验证级诊断：定位旧域OCR裁剪/读字退化，加入清晰旧域训练文字回放和教师一致性；实体部分检查普通原子框/类别及下游裁剪变化，评估普通原子冻结或独立文字精修分支；pair先保持冻结。先证明非零训练候选能在保留旧域同时改善小字及新整图，再给出新的正式训练命令。保留验证约束，不通过放宽验收或筛除难图制造提升。

完整机器报告 `runs/chemical_structure_balanced_audit_20261022/final_diagnosis.json`，权重一致性 `deployed_weight_comparison.json`，曲线 `retention_curves.png`。以下2026-10-21章节为训练前准备记录，正式训练现已结束。

## 2026-10-21 下一轮：清晰小字、尺度均衡与整图保留

新入口 `scripts/train_chemical_structure_balanced.sh`，正式训练由用户执行。初始化使用旧域表现较好的v2 bundle，而非发生回退的readable bundle。基础数据是已有清洁训练图，新增数据从相同训练源骨架重新绘制新字母数字组合，Arial/Times/Courier三种字体、自然灰度和彩色、三倍矢量绘图后面积降采样，配套原生高分辨率参考、独立SMILES/CXSMILES、完整文字框和锚点。新绘制的小字不分画布尺寸，均采用墨迹高度>=7、框高>=8、平均字符宽>=3.5、灰阶>=8及参考墨迹质量检查；该规则只适用于这批新绘图，不解禁旧硬二值化可疑图。尺寸规则是保守筛查，不保证每个像素级字形都无歧义，仍提供人工复查图。原val682/test659图及顺序保持，新增图只进入train，旧holdout骨架和随机字面串受保护。

`ScaleCharacterSampler` 均匀选择尺寸后再选字符，再选择含该字符的训练项，防止大图或常见词占据训练。OCR、实体/锚点和pair均启用；预检对256/512/768三个尺寸分别要求全部62字符，最少每字符5次，不能只检查总集合。OCR另加入每步新生成的随机字符辅助图（含不同字体、小字和自然灰度，排除所有已知验证/测试字面串），不虚构分子SMILES，也不把辅助文本当化学图真值。OCR初始设置为10%新图已知缩写回放、60%均衡化学文字、15%在线字符、15%既有清洁辅助文字。所有旧隔离样本继续排除。

`WholeGraphGuard` 使用完全预测的框/锚点/键、相同多尺度OCR和E/Z后处理，直接比较旧589图和当前682图的完整结构及字面标签；总体、每个尺寸、可展开已知缩写、未知字面串、无文字控制图以及这些类型与尺寸的交叉组，均不能低于本阶段初始正确率，不能用大图增益抵消小图退步，或以自由串改善掩盖已知词回退。OCR阶段固定结构网络，实体阶段固定pair/OCR，pair阶段固定实体/锚点/OCR。每次保存候选检查点时执行整图检查，不符合保留条件的更新不能成为best；符合后优先按完整新验证图正确数、旧验证正确数选择，局部F1/GT框识字等只作同分比较，初始模型始终可回退。基线、选模协议版本和输入签名写入last.pth供断点恢复。正式检查使用完整两套val；smoke每域9张，按三尺寸和三类型分层。该18图检查已和独立评估器逐组一致，冻结图和OCR标签两种缓存推理也一致。旧/新测试图不用于选模。

正式上限为OCR4000步、实体/锚点6000步、pair6000步，默认每500步验证，连续6次验证无选中改善则停止；保存实际完成步数和requested_steps。全量整图保留检查比此前局部F1选模更耗时。smoke仍各8步，检查实际权重更新、非训练模块冻结、检查点重载、均衡采样、选模保留和完整推理，不把8步当准确率结论。

准备与最终smoke已完成：2400张补充绘图中2341张通过独立化学/坐标审计，607张未满足原生可辨性条件而隔离，新增保留1734张。合并后训练2853张，来自684个源分子，4358种独立字符串组合（前轮890种），化学原图中的相邻字符对3793/3844。原生OCR区域按256/512/768为1948/1420/1940，三个尺寸各覆盖62字符，最少字符出现100/91/127次；原256px只有28个文字区域且缺11字符。val682/test659图像及GT逐文件哈希与原集合一致。所有OCR原生裁剪与主训练图标签一一对应，未加模糊或硬二值化；辅助在线字符排除验证/测试字面串及辅助集保留字体。

最终入口实测位于 `runs/chemical_structure_balanced_stratified_smoke_20261021`：三阶段各8步，均有真实梯度/权重更新，三个尺寸均实际抽到，OCR实际生成17个在线新辅助样本；非训练部分未改，保存重载一致。完整推理在新val六图、E/Z六图、旧val六图跑通，另外每次选模18图分层整图检查通过。OCR选8步，实体/pair选初始检查点，故不宣称8步后整图准确率提高。三阶段另在 `runs/chemical_structure_balanced_resume_smoke_20261021` 完成8→9步恢复、优化器更新、保留基线/采样计数恢复和选中检查点验证。23项相关单元检查通过；4096张普通图回放全量准备预检通过；正式训练尚未启动。总报告 `runs/chemical_structure_balanced_audit_20261021/final_report.json`，smoke汇总 `runs/chemical_structure_balanced_stratified_smoke_20261021/smoke_summary.json`，小字复查图 `dataset_text_balanced_20261021/small_training_review.png`（左原生，中放大，右高分辨率参考），隔离浏览 `data_quarantine/text_readability_20261021/review.html`。默认正式输出使用新目录 `runs/chemical_structure_balanced_stratified_formal_20261021`，不覆盖已完成训练。

```bash
cd /data/bo/works/pys/portable_mp123_epoch119_20261013
DEVICE=cuda:1 bash scripts/train_chemical_structure_balanced.sh smoke
# smoke通过后由用户执行，勿覆盖已有训练目录。
DEVICE=cuda:1 bash scripts/train_chemical_structure_balanced.sh formal > runs/chemical_structure_balanced_stratified_formal_20261021.log 2>&1
```

## 2026-10-21 可辨性正式训练检查

`runs/chemical_structure_readable_formal_20261020` 三阶段均完成12000步；OCR从1500步恢复后正常结束，选2000步；实体/锚点选12000步；pair选5500步。冻结模块、检查点保存重载检查均通过，最终三套验证及流程末尾Finished标记完整。旧OCR日志开头的IndexError是已修复并恢复的历史错误，不能据此判定最后训练失败。

固定682张新验证图、相同E/Z后处理，旧v2整图意义一致220/682（32.26%），本轮356/682（52.20%），提升19.94个百分点，改善165张、退步29张。这里未知标签按完整文字及附着图比较，可展开标签按严格展开SMILES比较。按256/512/768尺寸，新结果55/212、132/235、169/235，旧为49/212、78/235、93/235。大中图有改善，小图仍弱。旧589张验证反而458→410（77.76%→69.61%），改善13张、退步61张。90%目标未达到，保留实验状态，不自动切换notebook默认模型。

OCR同一批GT文字框：训练前863/1228（70.28%），选中模型880/1228（71.66%）。256px为161→155/378，512px327→337/426，768px375→388/424；已知缩写276→269/280，随机串587→611/948。相同训练预处理下训练1840/1840正确、验证918/1228（74.76%），仍有显著泛化差距。OCR最佳完整文字数在2000步，最终12000步仅771/1228，因此不应原样延长训练。实体文字框F1由93.84%到98.70%；无OCR的原子/键图一致从1001到1017/1271（78.76%→80.02%），局部高F1不等于整图90%。

本轮清洗按最小原生字号做硬筛查过于保守，造成尺度失衡：保留训练文字区域256/512/768分别28/652/1160（256仅1.52%），验证却是378/426/424。整体62字符覆盖不代表各尺度覆盖：256px实际仅51/62，缺少 `7KLTWYdlruw`。不能把全部小字视为坏数据；后续应继续隔离字形受损/不可辨图，另生成和保留清晰小字，以配对清晰参考和尺度均衡补足训练。当前保留自然灰度的方向有效，但这套清洗/采样/选模组合没有实现小字改善或旧域保持。

固定本轮实体/锚点/pair、换回旧v2 OCR，旧589图恢复到431正确；仍低于完整旧v2的458。故回退同时涉及OCR与实体/连接域适配，不是简单换一个OCR权重即可彻底解决。此消融仅验证，不改部署默认，不据独立测试选模。下一轮建议采用清晰小字均衡采样、更多独立在线字符串、按尺寸/已知词/自由串及旧域设保留约束，并采用更早停止；不继续原样堆步数。

固定模型和阈值、一次性新独立测试659图结果338/659（51.29%）：可展开标签154/198（77.78%），未知字面串184/461（39.91%）。256/512/768分别56/210（26.67%）、124/222（55.86%）、158/227（69.60%）。测试未用于训练、选模或阈值调节；它与旧336图测试不同，不能把两个集合的百分比直接当作本轮提升。独立测试与验证结论一致：中大图改善方向有效，小字和全图90%仍未解决。

报告位于 `runs/chemical_structure_readable_audit_20261021`：总报告 `final_diagnosis.json`、错误分类 `structure_diagnosis.json`、`initial_diagnosis.json`、`ocr_gap/summary.json`、`paired_new_validation.json`、`known_regression.json`、`known_diagnosis.json` 和 `old_val_previous_ocr/summary.json`；曲线为 `training_diagnosis.png`/PDF。最新三阶段bundle为 `runs/chemical_structure_readable_formal_20261020/pairs/bundle`，profile仅补记实验状态和评估，检查未更新网络权重。

## 2026-10-20 小字可辨性与可逆隔离

旧灰度绘图在缩小前后各进行一次阈值245二值化，容易把抗锯齿边缘一起涂黑，造成笔画粘连。新 `--readable-text` 绘图保留自然灰度/彩色与抗锯齿，取消这两次二值化，同时减少缩小和画布留白；不对旧图做插值“修复”，而是从分子和文字真值重新绘制。旧绘图默认行为保留用于复现。

`scripts/prepare_readable_text_data.py` 对训练图执行独立于OCR预测的保守筛查：旧灰度文字渲染来源、原生墨迹高度<12px、框高<14px、平均字符宽<5px、文字触及画布边界。整图及SMILES/CXSMILES/原子键真值一同索引到 `data_quarantine/text_readability_20261020`，相应裁剪的所有增强版本一并排除。隔离是预防措施，不等于已经证实每张图文字标错；该规则也不能证明所有保留字符均无遮挡/粘连。原图不移动不覆盖，val/test保持原样，不用筛掉难测试图的方式抬高分数。隔离文件使用硬链接节省磁盘，复查时视为只读；新旧训练清单完全分开。

已隔离旧字符训练图923张、旧缩写训练图1578张，连带OCR裁剪5538+9468张；辅助字符训练中2800张原生小字也单独归档。保留集合分别为 `dataset_text_characters_clean_20261020`、`dataset_text_training_clean_20261020`、`dataset_character_pretraining_clean_20261020`。旧文字数据清洗后可用训练词很少，正式入口改用重新绘制的 `dataset_text_readable_20261020` 及原生裁剪 `dataset_text_readable_crops_20261020`，不重新混入旧图；replay只取新图中已知缩写，不借助词表改写预测。

配对诊断固定v2模型，原验证集前36张有字图重新绘制，35张通过独立化学/坐标审计，共70个相同字面真值区域：旧图正确38/70，新图54/70；CER 24.23%→6.14%，19个改善、3个回退。256px组10/24→16/24，512px组12/24→19/24，768px组16/22→19/22。这里同时改变保灰度与缩放策略，使用GT文字框，仅为绘图/OCR诊断，不能当作整图分子正确率或单独二值化消融。原验证/测试文件不变，模型不更新。对照图：`runs/text_readability_audit_20261020/paired_validation/before_after.png`，详细数据同目录 `summary.json`。

新入口 `scripts/train_chemical_structure_readable.sh` 默认从v2最终bundle继续适配，训练前重查主图、裁剪来源、辅助字符与replay的质量清单和62字符覆盖；已知旧验证集的随机字面串不能进入新主训练集。OCR原生裁剪不追加模糊/对比度版本。三阶段仍为OCR、完整实体/连接锚点、pair；测试集不用于训练或选模；原589张验证也继续单列报告。notebook仍仅用于训练后的推理测试，不自动切换到smoke权重。

新一轮计划1200个源分子、3600个视图；209个视图未通过渲染/化学/坐标检查，3391个通过独立真值审计。只筛训练部分后保留1119张（651个源分子，256/512/768px分别81/390/648张），文字原生裁剪1840个，890种训练字符串（旧504种）、2418种相邻字符对（旧1709种），覆盖全部62字符。新训练图另隔离931张，原因计数允许重叠：墨迹/框高度不足927、平均字符宽不足366、贴边18、旧holdout随机字面串碰撞6。新val682/test659不按清晰程度删减；它们是新的合成评测集合，与旧344/336不可直接逐图对比。新train只用质量通过图；旧holdout继续作为独立回归检查。数据报告 `runs/text_readability_audit_20261020/data_summary.json`，隔离浏览 `data_quarantine/text_readability_20261020/review.html`。旧两套数据9272个图像/真值文件哈希全部未变，见同目录报告 `original_integrity.json`；正式4096个普通图回放预检也已通过。

三阶段各8步smoke已完成：OCR、decoder、anchor、pair均有实际梯度与权重更新；冻结部分未变化，保存重载一致，六图新验证/六图E/Z验证/六图旧验证推理均跑通。OCR选第8步，实体和pair选初始检查点，故只证明训练和完整推理流程可运行，不宣称smoke后整图准确率提升。14项相关单元检查通过。汇总 `runs/chemical_structure_readable_smoke_20261020/smoke_summary.json`，完整本轮报告 `runs/text_readability_audit_20261020/final_report.json`。正式训练尚未执行，notebook默认权重保持原选择。

```bash
cd /data/bo/works/pys/portable_mp123_epoch119_20261013
# Smoke不代表充分训练；正式训练由用户执行。
DEVICE=cuda:1 bash scripts/train_chemical_structure_readable.sh smoke
DEVICE=cuda:1 bash scripts/train_chemical_structure_readable.sh formal > runs/chemical_structure_readable_formal_20261020.log 2>&1
```

中断恢复：此次正式OCR在1500步之后的验证中因空字符串传入 `expand_abbreviation` 而抛出IndexError；已修复为空字符串返回未知原子 `[*]`。1500步 `ocr/last.pth` 含优化器、最佳权重和随机数状态，保留不变。OCR尚未完成且后两阶段尚未开始时，可使用下列命令恢复；程序会核验原数据/配置签名，从保存步之后继续，OCR日志采用追加模式。`RESUME_OCR` 只用于OCR中断，不用于覆盖已完成阶段或恢复后两阶段。

```bash
RESUME_OCR=1 DEVICE=cuda:1 bash scripts/train_chemical_structure_readable.sh formal >> runs/chemical_structure_readable_formal_20261020.log 2>&1
```

## 2026-10-20 第二轮：化学排版字符覆盖与连接强化

正式第二轮已完成，结果检查：三个阶段均12,000步，OCR选10,500步，实体/锚点选12,000步，pair选11,000步。新字符验证126/344（36.63%），封存的新独立测试现已固定方案评估116/336（34.52%）；其中未知字符串27/234（11.54%），可展开标签89/102（87.25%）。旧589图458正确，低于上一轮同样E/Z组装的467。90%目标未达到。

受控OCR诊断用相同训练预处理得到训练1826/1846正确（98.92%）、验证315/616正确（51.14%），故主要是泛化不足，而非单纯推理预处理。部署读字验证307/616，已见缩写142/144、未见随机词165/472；新数据共有504种训练字符串（含12种已知缩写），相邻字符对1709/3844。实体文字框F1从93.15%到98.69%，pair局部F1到99.37%，但完整连接图仅772/933（82.74%）。验证后期已明显平台，不建议原样增加步数。

完整报告 `runs/chemical_structure_v2_audit_20261020/final_diagnosis.json`，收敛曲线 `training_diagnosis_curves.png`，训练/推理受控OCR对比 `ocr_gap/summary.json`。下一轮优先增加独立/在线随机字符串和字符组合、提高小字可辨性、保留自然灰度/抗锯齿、统一裁剪规范与抑制过拟合；先在GT框未见词上验证，再验证整图，不把高局部F1当作整图90%。下方保留此次已完成训练的入口及准备记录。

二维 E/Z 后处理修复已在原589张验证图上独立对照：固定全部模型权重和OCR结果，严格展开结构正确数432→467（73.34%→79.29%），修复35、退步0。详见 `runs/stereo_validation_ablation_20261020/summary.json`。没有用新的字符测试集选择该修复。Notebook 已开启 `ENABLE_DOUBLE_STEREO=True`，也可关闭复现旧组装。原独立测试323/586对应修复前流程，不能当作修复后的测试成绩。

新数据 `dataset_text_characters_20261020` 有1,705张通过审查的整分子图：训练1,025、验证344、测试336，分别来自353、118、114个源分子。排除了上一轮1,000个源分子；本轮按骨架划分。既有缩写作为回放保留，随机字母数字整串跨集合不重叠（训练492、验证164、测试160种）。所有62字符都进入真实Indigo化学绘图训练标签；跨尺寸图像中最少字符出现98次，D出现155次、T出现113次，多尺寸/裁剪增强并非独立源分子。质量报告为数据目录下 `character_quality_audit.json`。

未知字符串只提供字面CXSMILES和含`*`的部分SMILES，明确设置 `expanded_truth_available=False`，不虚构化学展开。OCR选模改用包含文字及其附着位置的标签图一致性，并保留辅助字符CER约束。完整推理分别报告完整展开真值适用数、标签图一致数和混合任务正确数，不能把未知标签的部分SMILES相等当作识字正确。新字符测试集保持封存，训练脚本不执行测试预测。

第二轮在上一轮正式权重基础上分三阶段：OCR领域适配（70%新化学裁剪、20%辅助文字、10%旧缩写裁剪回放）；完整框与锚点继续训练（普通分子回放）；现有分辨率自适应pair head强化（GT锚点/预测节点混合、近邻负例、正例键型损失、普通分子回放和冻结教师蒸馏）。骨干与共享encoder继续冻结，无需重新从零训练。Pair验证还包含旧589张已知缩写验证图。各阶段按验证选择权重，允许保留起始模型；8步smoke并不代表已获得进一步准确率提升。

正式训练命令（不会覆盖第一轮）：

```bash
cd /data/bo/works/pys/portable_mp123_epoch119_20261013
DEVICE=cuda:1 bash scripts/train_chemical_structure_v2.sh formal > runs/chemical_structure_v2_formal_20261020.log 2>&1
```

默认三个阶段各12,000步，每500步全量验证；OCR batch16，其余batch4，普通图回放4,096张。最终bundle为 `runs/chemical_structure_v2_formal_20261020/pairs/bundle`。`validation/` 保留原立体组装评分，`validation_stereo/` 另报二维E/Z恢复结果，`validation_known/` 复核旧589张已知缩写验证图；均用验证图，测试集不解封。是否满足90%需训练后再看新的完整预测结果。

训练后 notebook 设置 `INFERENCE_PROFILE='complete_entities_v1'` 并把 `TRAINED_BUNDLE_DIR` 指向上述 `pairs/bundle`。新增 `double_stereo_512` 展示本轮立体组装修复，`free_characters_512` 用于训练后检查随机字符串的字面CXSMILES。Notebook不参与训练。

断点续训使用对应阶段脚本的 `--resume .../last.pth`，保持数据、初始bundle、batch、学习率和验证范围一致，`--steps`为目标总步数。例如pair阶段：

```bash
/home/user/miniconda3/envs/bl2m/bin/python scripts/train_text_pairs.py \
  --dataset dataset_text_characters_20261020 \
  --bundle runs/chemical_structure_v2_formal_20261020/entities/bundle \
  --out runs/chemical_structure_v2_formal_20261020/pairs \
  --resume runs/chemical_structure_v2_formal_20261020/pairs/last.pth \
  --steps 12000 --eval-every 500 --batch-size 4 --replay-limit 4096 \
  --device cuda:1 --run-kind formal
```

以下保留第一轮训练与数据说明，第二轮优先使用上方v2入口。

第二轮预检结果：三个阶段各8步真实优化、各阶段从8步恢复到9步、完整模型重载、普通回放4,096张预检、两种分辨率pair分支、34个单元测试全部通过。OMe/Boc和新增E/Z样例通过Notebook/CLI一致性及MOL/CXSMILES往返；同时修复了芳香[nH]结构用芳香键形式导出MOL导致重读失败的问题，改为标准Kekulé形式导出。汇总：`runs/chemical_structure_v2_smoke_20261020/smoke_summary.json`。正式第二轮长训未启动。8步OCR和pair未获验证增益时保留原权重，并未将其宣称为新模型效果提升。

训练入口独立于 notebook。Notebook 只负责训练后图片输入、预测可视化、分子组装以及真值比较。正式训练由用户启动。

## 训练内容

1. `finetune_chemical_ocr.py` 复用完整 EasyOCR 英文预训练识别器，以化学文字裁剪和 62 字符辅助数据进行 CTC 领域适配。不使用缩写词典强制纠字。保留预训练输出类别顺序，推理时仅允许英文字母和数字。
2. `train_complete_entities.py` 解冻原有 atom box decoder。完整字符串只有一个 `*` 框真值；字符串内部的单字母碎片没有独立目标，通过一对一匹配与分类损失学习抑制。其余普通原子仍有完整监督。每批约一半图片来自原普通分子训练集回放。
3. 同时训练 native ROI 像素特征和连接锚点分支，用独立 renderer-layout 锚点监督。共享 backbone、encoder、原 line decoder 和已训练 pair head 冻结。原有 native CTC 分支不用于最终读字，避免用随机训练字符头替换预训练 OCR。

OCR 按验证集真值骨架上的化学身份选择 checkpoint，并设辅助字符 CER 保持约束；完整实体按验证集普通原子和文字框 F1 均值选择，同分时比较锚点误差，普通原子 F1 相对初始最多下降 0.01。允许保留初始权重，不能把“完成优化步数”当作“获得提升”。整图正确率由随后独立的完整预测评估报告。

## 已准备数据

`dataset_text_training_20261019`：从 1,000 个源分子计划生成 3,000 张图，69 个渲染/真值校验失败样本排除，最终 2,931 张有效图：

|集合|图片|实际保留源分子|完整文字区域|
|---|---:|---:|---:|
|train|1,756|596|3,158|
|val|589|200|1,060|
|test|586|198|1,054|

原图包含 256、512、768 三种尺寸，949 张彩色，其余灰度。每张先有独立 SMILES、字面非展开 CXSMILES、普通原子框、完整文字框、键和锚点真值，再做预测。源分子/原 Murcko 骨架跨 split 不重叠，SHA256 冻结图像与真值；普通分子回放也排除新旧文字验证/测试骨架。多尺寸不是独立源分子。

化学文字包括 12 种训练缩写，NHCbz/TBDMS 留出在测试；因此该集检验化学缩写，不能代表任意字母组合的化学绘图。另有 `dataset_character_pretraining_expanded_20261019`：6,200/620/620 个 train/val/test 字符样本，覆盖 62 字符，整串跨集不重复，包含小字号、颜色、重复字符及留出字体。任意字符样本只做文字监督，不虚构其展开化学意义。

`dataset_text_training_crops_20261019` 保存化学 OCR 裁剪与增强；仅训练集增强，验证/测试保持原图。数据来自旧 pair corpus 的 val 来源；本轮新增任务划分清楚，但最早 epoch119 的数据重叠情况未知。完整结构泛化还需保留外部新来源验证。

## 正式训练

```bash
cd /data/bo/works/pys/portable_mp123_epoch119_20261013
DEVICE=cuda:1 bash scripts/train_chemical_structure.sh formal > runs/chemical_structure_formal_20261019.log 2>&1
```

默认 OCR 12,000 步（batch 16），完整实体/锚点 12,000 步（batch 4），每 500 步全量验证，回放 4,096 张普通分子图。使用 bl2m 的绝对 Python 路径。修改 `DEVICE` 可换 GPU，`RUN_DIR` 可指定新输出目录；已有 checkpoint 的目录不会被重置。

流程最后在全部 589 张验证图上分别评估“网络框＋锚点＋适配 OCR”和“额外文字区域确认＋锚点＋适配 OCR”，全部用预测节点和预测连接，输出 `validation_network/summary.json` 与 `validation_hybrid/summary.json`。可按验证结果选择推理方案，再进行一次固定方案的测试集评价。训练中实体模块不读取测试标签；OCR 在验证选定权重后另报告测试读字诊断，不能拿它作为整图测试成绩。

日志位于运行目录的 `ocr.log`、`entities.log` 以及两个验证日志。模型存于 `entities/bundle/{base,pair,anchor,ocr}.pth`。原 encoder 不变，但 atom decoder 已改变，所以完整 base 哈希也会改变；打包时先验证冻结模块逐张量一致，再更新 pair/anchor 的兼容性元数据，加载时仍严格校验新哈希。

## 断点续训

两阶段每次验证都原子写入 `last.pth`，包含优化器、随机状态、最佳验证权重和输入签名。阶段内数据、batch、学习率和验证范围需保持一致。`--steps` 是目标总步数，应大于已完成步数。

OCR 阶段示例：

```bash
/home/user/miniconda3/envs/bl2m/bin/python scripts/finetune_chemical_ocr.py \
  --dataset dataset_text_training_20261019 --crops dataset_text_training_crops_20261019 \
  --aux dataset_character_pretraining_expanded_20261019 \
  --out runs/chemical_structure_formal_20261019/ocr \
  --resume runs/chemical_structure_formal_20261019/ocr/last.pth \
  --steps 12000 --eval-every 500 --device cuda:1 --run-kind formal
```

完整实体阶段示例（OCR 阶段完成后）：

```bash
/home/user/miniconda3/envs/bl2m/bin/python scripts/train_complete_entities.py \
  --out runs/chemical_structure_formal_20261019/entities \
  --ocr-checkpoint runs/chemical_structure_formal_20261019/ocr/best.pth \
  --resume runs/chemical_structure_formal_20261019/entities/last.pth \
  --steps 12000 --batch-size 4 --eval-every 500 --replay-limit 4096 \
  --device cuda:1 --run-kind formal
```

若 OCR 已结束而实体阶段尚未开始，删除上述实体命令中的 `--resume` 一行即可单独启动实体阶段。之后可用 `evaluate_structure_backends.py --dataset dataset_text_training_20261019 --bundle .../entities/bundle --split val --network-entities-only --adapted-only --out ...` 继续验证。

## 训练后 notebook

在 `notebooks/full_inference.ipynb` 配置单元设置：

```python
INFERENCE_PROFILE = 'complete_entities_v1'
TRAINED_BUNDLE_DIR = PROJECT_ROOT/'runs/chemical_structure_formal_20261019/entities/bundle'
```

此模式直接使用训练后的网络框、锚点和 OCR。若验证证明外部文字区域确认仍有收益，可在配置单元末尾显式设置 `ENABLE_TEXT_AWARE_ATOMS = True`、`TEXT_REGION_USE_SELECTED_OCR = True`、`TEXT_REGION_DETECTOR_ONLY = True` 和 `ALLOW_UNMAPPED_TEXT = True` 复核混合流程。Notebook 不含训练步骤。

## Smoke 的含义

`bash scripts/train_chemical_structure.sh smoke` 只做 8 步 OCR 和 8 步实体/锚点训练，限制验证子集，并执行两个完整推理路径；另检查断点续训。检查真实非零梯度、权重更新、有限损失、冻结参数不变、保存/重载一致以及键头兼容性。8 步不是充分训练，子集结果不能用来宣称达到 90% 整图准确率。

2026-10-19 已通过上述 smoke、两个阶段从第 8 步恢复到第 9 步、正式 batch=4/4,096 张回放的额外短测、25 个单元测试，以及加载新 bundle 的 notebook/CLI 一致性和 MOL/CXSMILES 往返。机器可读汇总为 `runs/chemical_structure_smoke_20261019/smoke_summary.json`。

2026-10-20 更新：正式训练已于10月19日17:02 UTC连同验证流程全部完成。OCR最终选第10,000步，完整实体/锚点选第12,000步。固定验证最优流程在独立测试集严格结构正确323/586（55.12%），尚未达到90%。同批验证新模型432/589、旧原型202/589；详情见 `runs/chemical_structure_audit_20261020/final_diagnosis.json`。当前 notebook 默认已切换正式 bundle，并在512尺寸OMe/Boc样例通过完整图、OCR、CLI一致性和MOL/CXSMILES往返回归。不要直接重跑上方正式命令覆盖已有目录。
