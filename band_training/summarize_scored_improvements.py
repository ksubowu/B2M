"""Reviewable validation development report, including rejected approaches."""
import csv,json,hashlib,shutil,zipfile
from pathlib import Path


def main():
    root=Path('runs/scored_bond_improvements_20261119');previous=Path('runs/scored_bond_v3_comparison_20261118')
    read=lambda p:json.loads(p.read_text())
    roots=dict(old_original=Path('runs/full_v3_graph_metrics_20261117'),old_pair_dedup=previous/'old_pair_control',
               new_original=previous,new_postprocess=root/'selected')
    all_counts={};table=[]
    for name,path in roots.items():
        c=read(path/'prediction_cache/report.json')['counts'];f=read(path/'graph_ocr_verified/functional_report.json')['groups']['all']
        tp,fp,fn=(c['bond_'+k] for k in ['tp','fp','fn']);n=c['images']
        table.append(dict(model=name,images=n,atom_f1=2*c['atom_tp']/(c['atom_pred']+c['atom_gt']),charge_f1=2*c['charge_tp']/(c['charge_pred']+c['charge_gt']),
            bond_precision=tp/(tp+fp),bond_recall=tp/(tp+fn),bond_f1=2*tp/(2*tp+fp+fn),bond_tp=tp,bond_fp=fp,bond_fn=fn,
            atom_exact=c['atom_exact'],charge_owner_exact=c['charge_owner_exact'],bond_exact=c['bond_exact'],full_exact=c['legacy_full_exact'],
            graph_star_exact=f['graph_star_exact'],graph_ocr_exact=f['graph_ocr_exact']))
        all_counts[name]=c
    ablations=read(root/'followup_ablations/report.json');dense=read(root/'dense_line_diagnosis.json')
    config=dict(checkpoint=str(Path('runs/scored_bond_v3_formal_20261117/best_joint.pt').resolve()),
        checkpoint_sha256=read(root/'candidate_cache/report.json')['checkpoint_sha256'],line_threshold=.25,band_threshold=.2,
        line_association='forward',refine_line_peaks=True,stereo_from_band=True,
        neighborhood_refinement='One segment per original line peak; inspect its 3x3 neighbor cells only. Candidate line score >=max(threshold,0.5*peak score); require >=1 pixel reduction in endpoint box-boundary +0.02 center-distance cost. Peak presence score/query identity retained; geometry source separately recorded.',
        stereo='For wedge/dash only, align line endpoint order with ordered band quad axis; retain positions.',
        evaluation_scope='Selected on the existing validation set; four-image gain over old+dedup is not an independent-test improvement claim.')
    out=root/'review';out.mkdir(exist_ok=True)
    report=dict(status='completed',images=1498,metrics=table,recommended_config=config,
        ablations=ablations['results'],gt_geometry=ablations['ground_truth_geometry'],dense_diagnosis=dense['counts'],
        findings=['Lowering line threshold added false positives without recovering true typed edges.',
            'Disabling line local peaks recovered 135 true typed edges but increased FP from 333 to 1053, reducing exact graphs.',
            'Exclusive assignment reaches all bonds exact on GT at cluster radius 0/2/4 but degrades noisy prediction assembly; rejected for production predictions.',
            'Restricted endpoint refinement plus band orientation and line threshold .25 improves exact graphs from 1363 to 1381, versus old+dedup 1377.',
            'No weights were changed in these postprocessing comparisons; OCR backend and split unchanged.'],
        next_training='Line-only neighborhood supervision, ordered relative endpoint and direction loss; atom/charge/shared/band frozen. Train-only complexity/directed-bond sampling. Smoke result and separate formal progress determine training status.',
        training_paths=dict(smoke='runs/line_neighborhood_smoke_v2_20261119',formal='runs/line_neighborhood_formal_v2_20261119'),
        caveats='Validation development results; isotope/implicit H/CIP/E-Z not independently scored. Functional-only dummy policy and fixed EasyOCR preserved. All GT inputs are restricted to scoring or explicitly separate oracle controls.')
    smoke=Path(report['training_paths']['smoke'])/'results.json'
    if smoke.exists():report['training_smoke']=read(smoke)
    progress=Path(report['training_paths']['formal'])/'progress.json'
    if progress.exists():report['formal_progress_at_report_generation']=read(progress)
    write=lambda path,obj:path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
    write(out/'report.json',report);write(out/'recommended_inference.json',config)
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    with (out/'ablations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['variant','full_exact','full_rate','bond_tp','bond_fp','bond_fn','fixed','regressed']);w.writeheader()
        for r in ablations['results']:
            c=r['counts'];w.writerow(dict(variant=r['variant'],full_exact=c['legacy_full_exact'],full_rate=r['full_exact_rate'],
                bond_tp=c['bond_tp'],bond_fp=c['bond_fp'],bond_fn=c['bond_fn'],fixed=r['fixed'],regressed=r['regressed']))
    lines=['双 score 改善实验：相同1498张验证图，权重未改变。','',
           '模型 | 严格全图 | 键全对 | graph* | graph_ocr | 键FP/FN']
    for r in table:
        values=' | '.join(f"{r[k]}/1498 ({100*r[k]/1498:.2f}%)" for k in ['full_exact','bond_exact','graph_star_exact','graph_ocr_exact'])
        lines.append(f"{r['model']} | {values} | {r['bond_fp']}/{r['bond_fn']}")
    lines+=['','推荐参数：line=0.25、band=0.2、forward KD-tree、峰值邻域受限端点修正、band辅助有向键端序。',
            '同一验证集开发选择；优于旧权重去重的4张差距很小，需要独立数据验证。',
            'GT一对一匹配可以100%恢复键，但直接用于预测端点明显下降，未启用。',
            '降低line阈值没有追回TP；移除局部峰值筛选追回135条键，却产生过多FP，未启用。',
            '后续训练：仅line head微调。请查看对应smoke/results.json与formal/progress.json确认实时状态。',
            '', '推理命令：',
            '/home/bo/bo/miniconda3/envs/bb2m/bin/python -m band_training.predict_scored_bonds --checkpoint runs/scored_bond_v3_formal_20261117/best_joint.pt --image IMAGE --output OUTPUT --line-threshold .25 --band-threshold .2 --refine-line-peaks --stereo-from-band']
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    sources=out/'source';sources.mkdir(exist_ok=True)
    for name in ['scored_bond_model.py','scored_bond_postprocess.py','predict_scored_bonds.py','cache_scored_candidates.py',
                 'evaluate_scored_improvements.py','line_neighborhood_loss.py','finetune_line_neighborhood.py','train_scored_bonds.py',
                 'test_line_neighborhood.py','test_scored_bonds.py','summarize_scored_improvements.py']:
        shutil.copyfile(Path(__file__).with_name(name),sources/name)
    write(out/'source_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.iterdir()})
    with zipfile.ZipFile(root/'review.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(root))
    print('\n'.join(lines))


if __name__=='__main__':main()
