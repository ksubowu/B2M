"""Select a unique audited replay mixture using training-only prior constraints."""
import argparse,json,random,math,csv
from pathlib import Path
from collections import Counter
from .continue_indigo import write,sha


def features(m):
    return {**{'atom_'+str(k):int(m['atom_classes'].get(str(k),0)>0) for k in [8,9,10,11,12]},
            **{'bond_'+str(k):int(m['bond_classes'].get(str(k),0)>0) for k in [2,3,4,5]}}


def select(records,metadata,size,base_ids,seed=20261125,multiplier=1.25):
    base=[r for r in records if r['sample_id'] in base_ids];new=[r for r in records if r['sample_id'] not in base_ids]
    rng=random.Random(seed);rng.shuffle(new);counts=Counter()
    for r in base:counts.update(features(metadata[r['sample_id']]))
    reference=dict(counts);caps={k:math.ceil(v/len(base)*size*multiplier) for k,v in counts.items()};selected=list(base);skipped=Counter()
    for row in new:
        f=features(metadata[row['sample_id']]);violated=[k for k,v in f.items() if counts[k]+v>caps[k]]
        if violated:skipped.update(violated);continue
        selected.append(row);counts.update(f)
        if len(selected)==size:break
    assert len(selected)==size,('Insufficient feasible new images',len(selected),caps)
    assert all(counts[k]<=caps[k] for k in caps)
    assert len({r['sample_id'] for r in selected})==len(selected)
    return selected,dict(original_images=len(base),new_images=size-len(base),constraint_multiplier=multiplier,reference_feature_images=reference,selected_feature_images=dict(counts),caps=caps,rejected_constraints=dict(skipped))


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',type=Path,required=True);p.add_argument('--base-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    records=json.loads((a.preflight/'train.json').read_text());meta={m['sample_id']:m for m in json.loads((a.preflight/'assets.json').read_text())}
    ids={r['sample_id'] for r in json.loads(a.base_manifest.read_text())};chosen,stats=select(records,meta,32768,ids)
    val=json.loads((a.preflight/'val.json').read_text());assert not ({r['group'] for r in chosen}&{r['group'] for r in val})
    manifest=a.output/'controlled_train.json';write(manifest,chosen)
    plan=dict(status='passed',train_manifest=str(manifest.resolve()),train_manifest_sha256=sha(manifest),train_images=len(chosen),validation_images=len(val),selection=stats,
        controls=dict(init='runs/line_neighborhood_formal_v2_20261119/best_joint.pt',epochs=3,batch_size=16,updates_per_epoch=2048,total_image_exposures=98304,seed=20261124,
            freeze_backbone_encoder=True,box_lr=2e-6,shared_bond_lr=3e-6,line_lr=5e-6,band_lr=5e-6,warmup_updates=200,patience=0),
        arms=dict(prior_heads='Train atom/charge box + bond heads on controlled prior mixture',prior_bond_only='Identical mixture and budget; box decoder frozen in eval mode; bond heads only'),
        scope='Selection references original TRAINING image-presence rates only, capped at 1.25x. Validation is used solely for overlap exclusion and frozen evaluation. All records come unchanged from the passed preflight; no duplicate image entries, no source edits. Previous replay_heads remains the equal-budget prior-uncontrolled comparison.')
    write(a.output/'experiment_plan.json',plan)
    with (a.output/'selected_prior.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['feature','original_rate','selected_rate','cap_rate']);w.writeheader()
        for k,v in stats['reference_feature_images'].items():w.writerow(dict(feature=k,original_rate=v/stats['original_images'],selected_rate=stats['selected_feature_images'][k]/len(chosen),cap_rate=stats['caps'][k]/len(chosen)))
    print(json.dumps(plan,indent=2))


if __name__=='__main__':main()
