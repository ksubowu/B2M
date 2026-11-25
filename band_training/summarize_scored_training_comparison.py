"""Summarize frozen v2, v2 pair-dedup control and independently scored v3."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    runs = Path('runs')
    root = runs / 'scored_bond_v3_comparison_20261118'
    roots = dict(old=runs/'full_v3_graph_metrics_20261117',
                 old_pair_control=root/'old_pair_control', new=root)
    read = lambda p: json.loads(p.read_text())
    reports = {k: read(p/'prediction_cache/report.json') for k,p in roots.items()}
    functions = {k: read(p/'graph_ocr_verified/functional_report.json') for k,p in roots.items()}
    metrics = {}
    for key, report in reports.items():
        c = report['counts']; n = c['images']; m = {}
        for name in ['atom', 'charge']:
            m[name+'_precision'] = c[name+'_tp']/c[name+'_pred']
            m[name+'_recall'] = c[name+'_tp']/c[name+'_gt']
            m[name+'_f1'] = 2*c[name+'_tp']/(c[name+'_pred']+c[name+'_gt'])
        tp,fp,fn = (c['bond_'+k] for k in ['tp','fp','fn'])
        m.update(bond_precision=tp/(tp+fp),bond_recall=tp/(tp+fn),bond_f1=2*tp/(2*tp+fp+fn))
        for name in ['atom_exact','charge_detection_exact','charge_owner_exact','bond_exact','legacy_full_exact']:
            m[name] = c[name]/n
        for name in ['graph_star_exact','graph_ocr_exact']:
            m[name] = functions[key]['groups']['all'][name]/n
        m['charged_owner_exact'] = c['charged_owner_exact']/c['charged_images']
        alias = functions[key]['groups']['with_alias']
        m['alias_star_exact'] = alias['graph_star_exact']/alias['images']
        m['alias_ocr_exact'] = alias['graph_ocr_exact']/alias['images']
        m['ocr_given_correct_alias_graph'] = alias['graph_ocr_exact']/alias['graph_star_exact']
        metrics[key] = m
    manifests = {}
    for split in ['train','val']:
        paths = [runs/name/(split+'.csv') for name in ['full_v3_paired_formal_20261117','scored_bond_v3_formal_20261117']]
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        assert len(set(hashes)) == 1
        manifests[split] = hashes[0]
    sample_scores = {k: {s['sample_id']:s for s in read(p/'prediction_cache/component_samples.json')} for k,p in roots.items()}
    assert all(set(s)==set(sample_scores['old']) for s in sample_scores.values())
    transitions = {}
    for key in ['old','old_pair_control']:
        transitions[key+'_to_new'] = {
            metric: dict(fixed=sum(not sample_scores[key][sid][metric] and sample_scores['new'][sid][metric] for sid in sample_scores[key]),
                         regressed=sum(sample_scores[key][sid][metric] and not sample_scores['new'][sid][metric] for sid in sample_scores[key]))
            for metric in ['atom_exact','charge_owner_exact','bond_exact','legacy_full_exact']}
    rows = [dict(metric=name,**{k:metrics[k][name] for k in roots},
                 new_minus_old_pp=100*(metrics['new'][name]-metrics['old'][name]),
                 new_minus_pair_control_pp=100*(metrics['new'][name]-metrics['old_pair_control'][name])) for name in metrics['new']]
    report = dict(status='completed',training=dict(epochs=20,steps=40180,seconds=50646.56015133858,
        ended_hong_kong='2026-11-18 10:05',exit_code=0,best_epoch=16),
        images=1498,manifest_sha256=manifests,metrics=rows,sources=reports,functional_graph_metrics=functions,
        image_transitions=transitions,independent_head_metrics=read(runs/'scored_bond_v3_formal_20261117/epoch_16_metrics.json'),
        scope='Same 1498 validation images and unchanged splits. Old pair control changes only postprocessing, without retraining. All table metrics recomputed from frozen inference caches; training-time floating point kernels can cause 1-2 candidate differences, with best-epoch strict graph count reproduced exactly. Dummy graphs anonymize aliases/R-sites only, preserving real unsupported elements. OCR is fixed EasyOCR English g2, no GT lexicon. Exact graph isomorphism, not partial MCS; isotope/implicit H/CIP/E-Z are not independently scored.',
        conclusion='v3 improves over original v2, but does not beat the frozen v2+pair-dedup control on exact molecular graphs. Do not attribute all gains to independent heads.')
    out=root/'comparison';out.mkdir(exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['Training completed 2026-11-18 10:05 HKT; 20 epochs; best checkpoint epoch 16.',
           'Same 1498-image validation set. Columns: old / old+pair dedup / new / new-old pp / new-control pp.','']
    lines += [r['metric']+': '+ ' / '.join(f'{100*r[k]:.4f}%' for k in roots)+f" / {r['new_minus_old_pp']:+.4f} / {r['new_minus_pair_control_pp']:+.4f}" for r in rows]
    lines += ['',report['conclusion'],'',report['scope']]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
