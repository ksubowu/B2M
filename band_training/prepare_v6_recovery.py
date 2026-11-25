"""Matched-budget, unique-image development experiments from audited v6 assets."""
import argparse,json,random,csv
from collections import Counter
from pathlib import Path
from .continue_indigo import sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',type=Path,required=True);p.add_argument('--base-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    train=json.loads((a.preflight/'train.json').read_text());val=json.loads((a.preflight/'val.json').read_text());meta={x['sample_id']:x for x in json.loads((a.preflight/'assets.json').read_text())}
    ids={x['sample_id'] for x in json.loads(a.base_manifest.read_text())};base=[r for r in train if r['sample_id'] in ids];new=[r for r in train if r['sample_id'] not in ids]
    rng=random.Random(20261124);rng.shuffle(base);rng.shuffle(new);size=32768
    nbase=round(size*len(base)/len(train));arms={'uniform_heads':base[:nbase]+new[:size-nbase],'replay_heads':base+new[:size-len(base)]}
    groups={'original':base,'new_pubchem':new,'validation':val};stats={};rows=[]
    for name,records in groups.items():
        counts=Counter();images=Counter();bonds=Counter()
        for r in records:
            d=meta[r['sample_id']];counts.update(d['atom_classes']);images.update({k:int(v>0) for k,v in d['atom_classes'].items()});bonds.update(d['bond_classes'])
        stats[name]=dict(images=len(records),atoms=sum(counts.values()),atom_counts=dict(counts),atom_image_counts=dict(images),bond_counts=dict(bonds))
        for label,count in counts.items():rows.append(dict(source=name,label=label,count=count,atom_fraction=count/sum(counts.values()),image_fraction=images[label]/len(records)))
    plans={}
    for name,records in arms.items():
        assert len(records)==size and len({r['sample_id'] for r in records})==size
        assert not ({r['group'] for r in records}&{r['group'] for r in val})
        write(a.output/(name+'_train.json'),records)
        plans[name]=dict(images=size,original=sum(r['sample_id'] in ids for r in records),new=sum(r['sample_id'] not in ids for r in records),sha256=sha(a.output/(name+'_train.json')))
    plan=dict(status='passed',arms=plans,source_distribution=stats,controls=dict(init='line_neighborhood_formal_v2_20261119/best_joint.pt',epochs=3,batch_size=16,updates_per_epoch=2048,images_seen_per_arm=98304,seed=20261124,backbone_encoder='frozen throughout',box_lr=2e-6,shared_bond_lr=3e-6,line_lr=5e-6,band_lr=5e-6,warmup_updates=200,sampling='Unique images uniformly shuffled in each arm; no repeats within an epoch. Replay arm includes all 13476 original training images.',validation_images=len(val),endpoint_and_ocr_rules='Unchanged'),scope='Development-set experiment, not held-out test. Fixed validation is never sampled into training. All subset records must exactly match the passed preflight. Full v6 source files unchanged.')
    write(a.output/'experiment_plan.json',plan)
    with (a.output/'source_class_distribution.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in plan.items() if k!='source_distribution'},indent=2))


if __name__=='__main__':main()
