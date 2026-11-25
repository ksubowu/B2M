"""Frozen atom FN/FP attribution; geometry pairing is diagnostic, not rescoring."""
import argparse,gzip,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from .continue_indigo import iou_matrix
from .mixed_data import ATOM_CLASSES


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    dataset=Path('/mnt/hdd2/bo/work/cvs/chemcv_data/real/indigo_full_v3_20261116');reports={};all_rows=[]
    for split in ['train','val']:
        totals=Counter();classes=defaultdict(Counter);confusion=Counter();rows=[];gt_shapes=defaultdict(list);fn_shapes=defaultdict(list)
        with gzip.open(a.root/split/'prediction_cache/predictions.jsonl.gz','rt') as f:
            for d in map(json.loads,f):
                atoms=d['atoms'];truth=d['ann']['atoms'];m={int(k):v for k,v in d['atom_mapping'].items()}
                missed=[i for i in range(len(truth)) if i not in set(m.values())];extra=[i for i in range(len(atoms)) if i not in m]
                for t in truth:
                    classes[t['category_id']].update(gt=1);gt_shapes[t['category_id']].append(min(t['bbox'][2:]))
                totals.update(images=1,atom_gt=len(truth),atom_tp=len(m),fn=len(missed),fp=len(extra),failed_images=int(bool(missed or extra)),
                    only_fn_images=int(bool(missed) and not extra),only_fp_images=int(bool(extra) and not missed),both_images=int(bool(missed and extra)))
                if not missed and not extra:continue
                raw=json.loads((dataset/d['record']['annotation']).read_text());gt_boxes=np.array([t['bbox'] for t in truth]);gc=gt_boxes[:,:2]+gt_boxes[:,2:]/2
                neighbors=np.linalg.norm(gc[:,None]-gc[None],axis=-1);np.fill_diagonal(neighbors,np.inf)
                near=neighbors.min(1) if len(truth)>1 else np.full(len(truth),np.inf)
                overlaps=iou_matrix([t['bbox'] for t in truth],[t['bbox'] for t in truth]);np.fill_diagonal(overlaps,0.)
                paired={};consumed=set()
                if missed and extra:
                    pb=np.array([atoms[i]['bbox'] for i in extra]);gb=gt_boxes[missed]
                    iou=iou_matrix(pb,gb);dist=np.linalg.norm((pb[:,:2]+pb[:,2:]/2)[:,None]-gc[missed][None],axis=-1)
                    diag=np.linalg.norm(gb[:,2:],axis=-1);valid=(iou>=.1)|(dist<np.maximum(8.,diag*.5))
                    cost=3*(1-iou)+dist/(4+diag[None]/2)
                    ii,jj=linear_sum_assignment(np.where(valid,cost,1e6))
                    for x,y in zip(ii,jj):
                        if valid[x,y]:paired[missed[y]]=dict(prediction=extra[x],iou=float(iou[x,y]),center_distance=float(dist[x,y]));consumed.add(extra[x])
                missing=[]
                for i in missed:
                    t=truth[i];cls=t['category_id'];r=raw['atoms'][i];pair=paired.get(i)
                    if pair is not None:
                        pred=atoms[pair['prediction']];reason='nearby_wrong_class' if pred['label']!=cls else 'residual_same_class_assignment'
                        confusion[f"{ATOM_CLASSES[cls]}->{ATOM_CLASSES[pred['label']]}"]+=1
                    else:reason='no_unmatched_prediction_with_accepted_geometry'
                    classes[cls].update(fn=1);classes[cls][reason]+=1;totals[reason]+=1
                    fn_shapes[cls].append(min(t['bbox'][2:]));totals['fn_unlabeled_vertex']+=int(not r.get('visible_label',True))
                    missing.append(dict(atom=i,label=cls,category=ATOM_CLASSES[cls],symbol=t['symbol'],bbox=t['bbox'],
                        visible_label=r.get('visible_label'),bbox_kind=r.get('bbox_kind'),nearest_gt_distance=float(near[i]) if np.isfinite(near[i]) else None,
                        gt_box_overlap=float(overlaps[i].max()),reason=reason,paired_prediction=pair))
                false=[]
                if atoms and truth:
                    cross=iou_matrix([p['bbox'] for p in atoms],[t['bbox'] for t in truth])
                for i in extra:
                    pred=atoms[i];classes[pred['label']].update(fp=1)
                    if i in consumed:reason='paired_to_unmatched_gt'
                    else:
                        j=int(cross[i].argmax());v=float(cross[i,j])
                        if v>=.1 and j in m.values():reason='extra_on_already_matched_atom'
                        else:reason='extra_not_overlapping_matched_atom'
                    totals[reason]+=1
                    false.append(dict(prediction=i,label=pred['label'],category=ATOM_CLASSES[pred['label']],bbox=pred['bbox'],score=pred['score'],reason=reason))
                row=dict(split=split,sample_id=d['sample_id'],record=d['record'],fn=len(missed),fp=len(extra),gt_atoms=len(truth),missing=missing,false_positives=false)
                rows.append(row);all_rows.append(row)
        assert totals['failed_images']==(677 if split=='train' else 59)
        reports[split]=dict(totals=dict(totals),classes={ATOM_CLASSES[k]:dict(v,fn_rate=v['fn']/v['gt'],
            gt_median_min_box=float(np.median(gt_shapes[k])),fn_median_min_box=float(np.median(fn_shapes[k])) if fn_shapes[k] else None) for k,v in classes.items()},confusions=dict(confusion))
        (a.output/(split+'_samples.json')).write_text(json.dumps(rows,indent=2))
    report=dict(splits=reports,scope='FN/FP reproduce existing label-aware matching. Residual predictions and unmatched GT are paired without classes under the SAME geometry gate for diagnosis only; near wrong-class pairing is an inferred explanation, not a changed metric. Raw decoder attribution and visual review are required to distinguish score, NMS and localization failures.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
