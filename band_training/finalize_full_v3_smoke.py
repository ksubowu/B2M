"""Collect the actual smoke/audit/full-validation evidence into a review bundle."""
import argparse,json,shutil,hashlib,zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);args=p.parse_args()
    run=args.run;read=lambda path:json.loads(path.read_text())
    smoke=read(run/'results.json');full=read(run/'full_validation/reloaded_metrics.json')
    baseline=read(run/'full_baseline_validation/reloaded_metrics.json');audit=read(args.audit/'report.json')
    order=read(args.audit/'target_order_report.json');assert order['status']=='passed'
    guard_path=run.parent/'full_v3_box_guard_smoke_v2_20261116/results.json'
    guard=read(guard_path) if guard_path.exists() else None
    bundle=run/'review';bundle.mkdir(exist_ok=True)
    strategy=dict(architecture='paired_band_line_v2',training_rows=16068,unique_train=13482,validation=1498,
        selected_postprocessing='forward',selection_reason='Measured full predicted validation graph accuracy exceeds exclusive_box_noisy',
        atom_index_supervision=False,endpoint_swap_minimum=False,charge_association='safe_adaptive',
        loss_weights=dict(box=1,bond=2,bond_exist_type=1,bond_center=1,bond_region=1,bond_line=5),
        planned_epochs=20,batch_size=8,feature_freeze_epochs=1,box_freeze_epochs=1,
        selection_guard=dict(max_atom_f1_drop=.005,max_charge_f1_drop=.03),
        learning_rates=dict(backbone=1e-6,encoder=2e-6,boxes=2e-6,band=5e-5,line=3e-4),
        model_outputs=dict(box_classes=18,atom_classes=13,charge_values=[-4,-2,-1,1,2],bond_classes=6,
            per_bond=['class and confidence','ordered quad 4x2','ordered line 2x2']),
        sampling='uniform shuffle of explicit train.csv (25% charge), no second sampler',
        endpoint_order='ordinary: left/top first; wedge/dash: preserve annotation direction; align quad ends to line',
        caveats=['Smoke weights, not a completed formal training run','Validation is used for selection, no independent test',
            'Metrics use the current 13 atom categories: pseudoatoms/unknown symbols map to other, no full label OCR or SMILES equivalence claim'])
    summary=dict(status='smoke_passed',strategy=strategy,audit=audit,target_order=order,smoke=smoke,
        full_validation_baseline=baseline,full_validation_after_smoke=full,box_protection_smoke=guard)
    (bundle/'report.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    percent=lambda v:f'{100*v:.2f}%'
    text=f'''全量数据的训练策略与 GPU smoke test（2026-11-16）

结论
训练链路通过，可开始有界的正式训练。已经从最新 atom/charge + band 权重热启动，新增与 band 同一检测位置绑定的 line head。
仅完成 smoke，不是全量训练完成。独立4图过拟合100步；主模型对完整16,068行训练清单shuffle后抽取128个batch训练（batch=4，共512条采样，显式重复保留）。
主模型没有使用4图过拟合得到的权重。初始及训练后模型均在完整1,498张冻结验证图上评测。

数据与坐标
14,980张唯一图（13,482训练，1,498验证），训练采样清单16,068行，其中带电4,017行（25%）。
全量344,768 atoms、368,318 bonds、3,119 charge boxes逐项通过检查，分子group及素材无训练/验证交叉。
图片RGB letterbox到640；boxes、quad、line共用实际四舍五入后的scale_xy和padding。当前只做颜色增强。
最大169个box、182条键，box query容量300，band候选不设160/300截断上限。

输出
1. atom/charge：18类检测框（13个atom类别 + 5个charge值类别），类别、置信度、box。
2. bond：每个检测实例输出6类键类别/置信度 + quad[4,2] + line[2,2]。
   band与line共享检测位置和类别，不需要后续猜测两份预测如何配对。
3. 后处理输出atom节点、电荷归属、bond begin/end、键型及未关联项；没有atom index/pointer训练目标。
   新image-only推理入口输出坐标恢复到原图的JSON以及原图|boxes|bands+lines三图横拼PNG。

固定方向
原band两端顺序与line相反的键有159,496/368,318（约43%），普通键的源atom编号不构成稳定的视觉方向。
训练适配器将普通键固定为左端在前、近竖直时上端在前；wedge/dash保留有向标注。quad两端对齐line。
源annotation未改写。loss不做交换端点取最小值；可用--endpoint-order source复现原始约定。
规范化后GT全部14,980张的键组装仍100%，原有电荷归属不变。

损失与计划
L = L_box_charge + 2*(L_exist_type + L_center + 1*L_region + 5*L_ordered_line)。
box使用原VFL/L1/GIoU及charge辅助监督；exist/type为热图focal，几何项为SmoothL1。
line为相对band中心的有序四坐标，按32px归一化，不接触预测/真实atom index。
正式配置20 epochs，batch=8（每epoch约2,009次更新），第1个epoch冻结backbone/encoder和box分支，此后低学习率解冻。
AdamW：backbone 1e-6、encoder 2e-6、box 2e-6、既有band 5e-5、新line 3e-4；warmup 200步、cosine衰减、梯度裁剪。
正式checkpoint选择增加检测保护：atom F1相对初始化最多下降0.005，charge F1最多下降0.03。
每epoch验证全部1,498张，按预测组图结果比较forward与容差一对一方案并选择对应checkpoint/推理方法。
电荷独立预测，组图使用safe_adaptive；不引入charge/atom pointer，不调用第三方OCR。

已测结果
4图line loss下降{percent(smoke['tiny_overfit']['relative_line_loss_reduction'])}；128步主训练无NaN，各训练分支及解冻特征均有非零有限梯度。
48张诊断子集有序端点误差6.00→4.85px（在GT检测位置读取，不等于端到端组图指标）。
128步训练及诊断耗时约{smoke['seconds']:.1f}秒，PyTorch峰值显存{smoke['peak_gpu_memory_gb']:.2f}GiB，不含外部进程占用。
全部权重tensor保存/重载一致，独立图像推理进程也已成功运行。

完整1,498张验证图（使用模型预测boxes/bands/lines）
                                    初始化         128步后
atom F1                             {percent(baseline['atom_f1'])}          {percent(full['atom_f1'])}
charge F1                           {percent(baseline['charge_f1'])}          {percent(full['charge_f1'])}
forward键 F1                        {percent(baseline['forward_baseline']['f1'])}          {percent(full['forward_baseline']['f1'])}
forward严格全图正确率               {percent(baseline['forward_full_graph_exact'])}          {percent(full['forward_full_graph_exact'])}
128步后，容差一对一方案键F1为{percent(full['bond_graph']['f1'])}、严格全图{percent(full['full_graph_exact'])}；本次推理权重选择forward。
GT一对一100%不能直接推导为有噪声预测也更优，后续每epoch继续跟踪两条路径。
注意：上述128步探索更新了box分支；charge真阳性54个保持不变，但误检14→21，F1出现下降。
因此最终正式策略改为首epoch冻结box，解冻后box LR降低到2e-6，并增加checkpoint检测保护。
另有32步冻结/解冻smoke记录在box_protection_smoke：前28步冻结box与features，最后4步解冻验证梯度；其48图子集指标不能替代全量验证。

评估范围
严格全图要求atom数量/分类匹配、所有typed bonds（wedge/dash有向）及电荷归属正确；未关联预测计错。
这是当前13种atom类别口径，不是canonical SMILES或完整字符标签正确率。
1,679个超出现有atom类别的符号/文本，涉及1,038张图，映射为other。它们的文字内容和缩写展开尚未训练。
正式训练尚未启动；smoke权重不覆盖旧latest权重或WebApp。

入口
训练：bash scripts/train_full_v3.sh
推理：python -m band_training.predict_full_v3 --checkpoint <run>/inference_joint.pt --image <image.png> --output <out_prefix>
smoke结果、全量验证逐样本记录、GT校验、源代码快照都保存在本run目录。
'''
    (bundle/'report.txt').write_text(text)
    for path in [args.audit/'report.json',args.audit/'target_order_report.json',run/'image_only_prediction.png',run/'image_only_prediction.json',run/'gt_alignment_01.png']:
        name='dataset_audit.json' if path==args.audit/'report.json' else path.name
        shutil.copyfile(path,bundle/name)
    source=bundle/'source';source.mkdir(exist_ok=True)
    hashes={}
    for name in ['train_full_v3.py','band_line_model.py','full_v3_data.py','model.py','geometry.py','endpoint_assignment.py','predict_full_v3.py','evaluate_full_v3.py','test_full_v3.py']:
        path=Path(__file__).with_name(name);shutil.copyfile(path,source/name);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (bundle/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    (bundle/'index.html').write_text('<meta charset="utf-8"><title>Full v3 smoke review</title><h1>训练策略与 smoke testing</h1><p><a href="report.txt">完整报告</a> | <a href="report.json">JSON结果</a></p><h2>GT alignment</h2><img width="100%" src="gt_alignment_01.png"><h2>Image-only prediction</h2><img width="100%" src="image_only_prediction.png">')
    with zipfile.ZipFile(run/'smoke_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in bundle.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(run))
    print(bundle.resolve())

if __name__=='__main__':main()
