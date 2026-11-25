"""Paired cache diagnostics. Ground truth is used only to attribute errors."""
import argparse,csv,gzip,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from .geometry import edge_key,target_edges,LineEndpointIndex


def load(path):
    with gzip.open(path,'rt') as f:return {r['sample_id']:r for r in map(json.loads,f)}


def diagnose(data):
    totals=Counter();reasons=Counter();confusion=Counter();samples=[];scores=defaultdict(list)
    for sid,d in data.items():
        mapping={int(k):v for k,v in d['atom_mapping'].items()};truth=target_edges(d['ann']);pred=Counter();by_pair={}
        for e in d['bonds']:
            a,b=mapping.get(e['begin'],-1),mapping.get(e['end'],-1)
            ok=min(a,b)>=0 and edge_key(a,b,e['label']) in truth
            for k in ['line_score','band_score']:scores[('bond_correct_' if ok else 'bond_wrong_')+k].append(e[k])
            if min(a,b)>=0:
                pred[edge_key(a,b,e['label'])]+=1;by_pair[tuple(sorted((a,b)))]=e
        heads=d['independent_heads'];lines=heads['lines'];raw_pairs=LineEndpointIndex([a['bbox'] for a in d['atoms']]).link_many([l['line'] for l in lines])
        pred_pairs={tuple(sorted((mapping.get(p[0],-1),mapping.get(p[1],-1)))) for p in raw_pairs if p is not None}
        oracle_pairs=LineEndpointIndex([a['bbox'] for a in d['ann']['atoms']]).link_many([l['line'] for l in lines])
        oracle={tuple(sorted(p)) for p in oracle_pairs if p is not None};available=set(mapping.values());miss=[]
        for edge,n in (truth-pred).items():
            pair=tuple(sorted(edge[:2]));e=by_pair.get(pair)
            if e is not None:reason='wrong_bond_class' if e['label']!=edge[2] else 'wrong_stereo_direction'
            elif not set(pair)<=available:reason='atom_detection_or_label_missing'
            elif pair in pred_pairs:reason='accepted_line_pair_present_but_no_correct_typed_bond'
            elif pair in oracle:reason='accepted_line_links_with_gt_boxes_only'
            else:reason='no_correct_pair_in_accepted_lines_even_with_gt_boxes'
            reasons[reason]+=n;miss.append(dict(edge=edge,count=n,reason=reason))
            if e is not None and e['label']!=edge[2]:confusion[f"{edge[2]}->{e['label']}"]+=n
        for i,a in enumerate(d['atoms']):scores['atom_matched_score' if i in mapping else 'atom_unmatched_score'].append(a['score'])
        metric=d['bond_metric'];totals.update(images=1,atom_fp=len(d['atoms'])-len(mapping),atom_fn=len(d['ann']['atoms'])-len(mapping),bond_fp=metric['fp'],bond_fn=metric['fn'])
        totals.update({k:int(v) for k,v in d['components'].items()})
        samples.append(dict(sample_id=sid,**d['components'],atom_fp=len(d['atoms'])-len(mapping),atom_fn=len(d['ann']['atoms'])-len(mapping),bond_fp=metric['fp'],bond_fn=metric['fn'],missing_bonds=miss))
    assert sum(reasons.values())==totals['bond_fn']
    return dict(counts=dict(totals),missing_bond_reasons=dict(reasons),bond_class_confusion=dict(confusion),
        score_distributions={k:dict(n=len(v),p10=float(np.quantile(v,.1)),median=float(np.median(v)),p90=float(np.quantile(v,.9))) for k,v in scores.items() if v},samples=samples)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    baseline=load(a.baseline);candidate=load(a.candidate);assert baseline.keys()==candidate.keys()
    x=diagnose(baseline);y=diagnose(candidate);transition={};paired=[]
    for component in ['atom_exact','charge_owner_exact','bond_exact','legacy_full_exact']:
        transition[component]=dict(regressed=sum(baseline[k]['components'][component] and not candidate[k]['components'][component] for k in baseline),fixed=sum(not baseline[k]['components'][component] and candidate[k]['components'][component] for k in baseline))
    for sid in baseline:
        b=baseline[sid];c=candidate[sid]
        paired.append(dict(sample_id=sid,**{'baseline_'+k:v for k,v in b['components'].items()},**{'candidate_'+k:v for k,v in c['components'].items()},baseline_bond_fp=b['bond_metric']['fp'],candidate_bond_fp=c['bond_metric']['fp'],baseline_bond_fn=b['bond_metric']['fn'],candidate_bond_fn=c['bond_metric']['fn']))
    a.output.mkdir(parents=True,exist_ok=True)
    for name,d in [('baseline',x),('candidate',y)]:
        (a.output/(name+'_samples.json')).write_text(json.dumps(d.pop('samples'),ensure_ascii=False))
    report=dict(baseline=x,candidate=y,transitions=transition,scope='Frozen validation-only diagnosis. FN attribution uses accepted thresholded lines, not dense network logits. GT boxes are a diagnostic counterfactual; no GT is used to repair predictions. Baseline and candidate use identical image IDs and scoring rules.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    with (a.output/'paired_samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)


if __name__=='__main__':main()
