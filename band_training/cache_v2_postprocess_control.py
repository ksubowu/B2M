"""Frozen v2 controls: isolate gains from pair dedup and v3 postprocessing."""
import argparse,gzip,json,copy
from pathlib import Path
from collections import Counter
from .continue_indigo import write
from .train import charge_exact
from .geometry import assemble_band_lines
from .train_scored_bonds import typed_metric
from .scored_bond_postprocess import assemble_scored_bonds
from .charge_association import assign_charges


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['pair_only','v3_post'],required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    counts=Counter();samples=[]
    with gzip.open(a.cache,'rt') as source,gzip.open(a.output/'predictions.jsonl.gz','wt',encoding='utf-8',compresslevel=3) as target:
        for line in source:
            d=json.loads(line);instances=d['instances'];mapping={int(k):v for k,v in d['atom_mapping'].items()}
            if a.mode=='pair_only':
                links={b['bond_instance']:b for b in d['bonds']};seen=set();keep=[]
                for i in sorted(links,key=lambda i:-instances[i]['score']):
                    e=links[i];pair=tuple(sorted((e['begin'],e['end'])))
                    if pair not in seen:seen.add(pair);keep.append(i)
                keep+=d['unlinked'];selected=[instances[i] for i in sorted(keep)]
                g=assemble_band_lines([r['bbox'] for r in d['atoms']],selected)
                g['untyped_lines']=[]
            else:
                decoded=dict(lines=[dict(query_id=i,line_score=p['score'],line=p['line']) for i,p in enumerate(instances)],
                    bands=[dict(query_id=i,band_score=p['score'],type_score=None,score=p['score'],label=p['label'],quad=p['quad']) for i,p in enumerate(instances)])
                g=assemble_scored_bonds([r['bbox'] for r in d['atoms']],decoded);selected=g['instances']
            score=typed_metric(g,d['ann'],mapping)
            assignments=assign_charges(d['atoms'],d['charges'],g['bonds'],method='safe_adaptive',lines=[p['line'] for p in instances])
            predictions=d['atoms']+d['charges'];charge=bool(charge_exact(predictions,d['atoms'],mapping,d['ann'],assignments))
            components=dict(d['components'],bond_exact=bool(score['exact']),charge_owner_exact=charge,
                legacy_full_exact=bool(d['components']['atom_exact'] and charge and score['exact']))
            counts.update(images=1,charged_images=int(bool(d['ann']['charges'])),atom_tp=len(mapping),atom_pred=len(d['atoms']),atom_gt=len(d['ann']['atoms']),
                charge_tp=len(d['charge_detection_mapping']),charge_pred=len(d['charges']),charge_gt=len(d['ann']['charges']),bond_tp=score['tp'],bond_fp=score['fp'],bond_fn=score['fn'],bands_pred=len(selected))
            counts.update({k:int(v) for k,v in components.items()})
            if d['ann']['charges']:counts.update(charged_detection_exact=int(components['charge_detection_exact']),charged_owner_exact=int(charge))
            else:counts.update(neutral_images=1,neutral_charge_exact=int(charge))
            updated=dict(d,instances=selected,bonds=g['bonds'],unlinked=g['unlinked_band_indices'],untyped_lines=g['untyped_lines'],
                charge_assignments=assignments,components=components,bond_metric=score)
            target.write(json.dumps(updated,ensure_ascii=False)+'\n');samples.append(dict(sample_id=d['sample_id'],**components))
    old=json.loads((a.cache.parent/'report.json').read_text())
    report=dict(status='completed',counts=dict(counts),dataset=old['dataset'],checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
        line_association='forward',checkpoint_epoch=20,previous_full_exact=None,previous_full_exact_reproduced=None,
        control=a.mode,score_proxy='Old v2 has no independent line score; its original bond score is used as line score for v3_post only.',weights_changed=False)
    write(a.output/'report.json',report);write(a.output/'component_samples.json',samples);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
