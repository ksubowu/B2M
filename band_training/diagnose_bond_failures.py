"""Frozen-cache bond error decomposition, counterfactuals and geometry diagnostics."""
import argparse,gzip,json,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .continue_indigo import write
from .geometry import edge_key,target_edges,aggregate,polygon_iou,assemble_band_lines
from .train_full_v3 import metric_row
from .full_v3_data import orient_targets

NAMES=['single','double','triple','aromatic','wedge','dash']


def paired_geometry(instances,quads,lines):
    """Label-free one-to-one diagnostic pairing, NOT the correctness metric."""
    if not instances or not quads:return {},{}
    q=np.asarray(quads);p=np.asarray([b['quad'] for b in instances]);pl=np.asarray([b['line'] for b in instances]);gl=np.asarray(lines)
    lengths=np.linalg.norm(gl[:,1]-gl[:,0],axis=1).clip(4)
    center=np.linalg.norm(p.mean(1)[:,None]-q.mean(1)[None],axis=-1)
    endpoints=np.minimum(np.linalg.norm(pl[:,None]-gl[None],axis=-1).mean(-1),np.linalg.norm(pl[:,None]-gl[None,:,::-1],axis=-1).mean(-1))
    iou=np.zeros(center.shape)
    for i in range(len(p)):
        for j in np.flatnonzero(center[i]<lengths*.8+8):iou[i,j]=polygon_iou(p[i],q[j])
    allowed=(iou>=.1)|((center<np.maximum(4,.2*lengths))&(endpoints<np.maximum(8,.6*lengths)))
    cost=1-iou+.1*center/lengths+.05*endpoints/lengths
    cost=np.where(allowed,cost,1e6);ri,gi=linear_sum_assignment(cost)
    pairs={int(i):int(j) for i,j in zip(ri,gi) if allowed[i,j]}
    details={i:dict(gt=j,iou=float(iou[i,j]),center_distance=float(center[i,j]),unordered_endpoint_error=float(endpoints[i,j]),
        gt_length=float(lengths[j])) for i,j in pairs.items()}
    return pairs,details


def row_from_pairs(instances,pairs,truth,mapping=None):
    pred=Counter();unlinked=0
    for p,pair in zip(instances,pairs):
        if pair is None:unlinked+=1;continue
        a,b=pair
        if mapping is not None:a,b=mapping.get(a,-1),mapping.get(b,-1)
        if min(a,b)<0 or a==b:unlinked+=1;continue
        pred[edge_key(a,b,p['label'])]+=1
    tp=sum((pred&truth).values())
    return dict(tp=tp,fp=len(instances)-tp,fn=sum(truth.values())-tp,unlinked=unlinked,exact=int(pred==truth and not unlinked),gt=sum(truth.values()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with gzip.open(a.cache,'rt') as f:data=[json.loads(l) for l in f]
    stats=Counter();fp=Counter();fn=Counter();buckets=defaultdict(list);class_gt=Counter();class_tp=Counter();class_pred=Counter();confusion=Counter();samples=[]
    oracles=defaultdict(list);thresholds=defaultdict(list);geometry=defaultdict(list);matched_stats=Counter();vertical=defaultdict(Counter)
    for d in data:
        ann=d['ann'];truth=target_edges(ann);mapping={int(k):v for k,v in d['atom_mapping'].items()};instances=d['instances']
        raw=json.loads((a.dataset/d['record']['annotation']).read_text());band=json.loads((a.dataset/d['record']['band_target']).read_text())
        from .mixed_data import letterbox_transform
        scale,offset,_=letterbox_transform(raw['width'],raw['height'])
        gtq=(np.asarray(band['quads']).reshape(-1,4,2)*scale+offset).tolist()
        gtl=(np.asarray([b['line'] for b in raw['bonds']]).reshape(-1,2,2)*scale+offset).tolist()
        gtq,gtl=orient_targets(band['labels'],gtq,gtl)
        gtinst=[dict(label=l,quad=q,line=line,score=1.) for l,q,line in zip(band['labels'],gtq,gtl)]
        match,matchinfo=paired_geometry(instances,gtq,gtl);gt_to_pred={j:i for i,j in match.items()}
        original=d['bond_metric'];n=len(ann['atoms']);b=len(ann['bonds'])
        stats.update(images=1,bonds_gt=b,bands_pred=len(instances),failed_images=int(not original['exact']),
            atom_correct_bond_failed=int(d['components']['atom_exact'] and not original['exact']),
            class_charge_correct_bond_failed=int(d['components']['atom_exact'] and d['components']['charge_owner_exact'] and not original['exact']))
        bucket='<=20 bonds' if b<=20 else '21-40 bonds' if b<=40 else '41-80 bonds' if b<=80 else '>80 bonds'
        buckets[bucket].append(original)
        for key,num in truth.items():class_gt[key[2]]+=num
        linked={x['bond_instance']:x for x in d['bonds']};remain=truth.copy();items=[];counts=Counter();pred=Counter()
        for i,item in enumerate(instances):
            label=item['label'];class_pred[label]+=1
            if i not in linked:category='unlinked_self_or_no_atoms';key=None
            else:
                edge=linked[i];u,v=mapping.get(edge['begin'],-1),mapping.get(edge['end'],-1)
                if min(u,v)<0:category='unmatched_or_wrong_class_atom';key=None
                else:
                    key=edge_key(u,v,label);pred[key]+=1
                    if remain[key]>0:category='correct';remain[key]-=1;class_tp[label]+=1
                    elif truth[key]>0:category='duplicate_same_typed_edge'
                    elif label in [4,5] and truth[edge_key(v,u,label)]>0:category='stereo_direction_reversed'
                    elif any({u,v}=={e[0],e[1]} for e in truth):category='wrong_bond_class'
                    else:category='wrong_atom_connection'
            info=dict(prediction=i,label=label,score=item['score'],category=category,mapped_edge=key,geometry_match=matchinfo.get(i))
            if category!='correct':fp[category]+=1;counts[category]+=1;items.append(info)
            q=np.array(item['quad']);line=np.array(item['line']);axis=q[2:].mean(0)-q[:2].mean(0)
            qlen=np.linalg.norm(axis);llen=np.linalg.norm(line[1]-line[0]);angle=math.degrees(math.atan2(abs(axis[1]),abs(axis[0])))
            which='unlinked' if i not in linked else 'linked'
            geometry[which+'_line_lengths'].append(llen);geometry[which+'_line_to_band_length'].append(llen/max(1e-5,qlen))
            vertical['near_vertical' if angle>=80 else 'other'].update(predictions=1,unlinked=int(i not in linked))
            if i in match:
                j=match[i];matched_stats.update(matched_instances=1,class_correct=int(label==band['labels'][j]))
                confusion[band['labels'][j],label]+=1
                geometry[which+'_matched_line_ratio'].append(llen/max(1e-5,np.linalg.norm(np.array(gtl[j])[1]-np.array(gtl[j])[0])))
                geometry[which+'_matched_endpoint_error'].append(matchinfo[i]['unordered_endpoint_error'])
        missing=truth-pred;missitems=[]
        for key,num in missing.items():
            candidates=[j for j,t in enumerate(ann['bonds']) if edge_key(t['begin'],t['end'],t['category_id']-1)==key]
            for _ in range(num):
                j=candidates.pop(0);samepair=[i for i,e in linked.items() if {mapping.get(e['begin'],-1),mapping.get(e['end'],-1)}=={key[0],key[1]}]
                if any(instances[i]['label']!=key[2] for i in samepair):reason='class_confusion_on_correct_atom_pair'
                elif samepair:reason='stereo_direction_on_correct_pair'
                elif j not in gt_to_pred:reason='no_geometrically_matched_region_at_threshold'
                else:
                    i=gt_to_pred[j]
                    if i not in linked:reason='paired_region_but_line_unlinked'
                    elif min(mapping.get(linked[i]['begin'],-1),mapping.get(linked[i]['end'],-1))<0:reason='paired_region_but_atom_detection_error'
                    else:reason='paired_region_but_wrong_endpoint_connection'
                fn[reason]+=1;missitems.append(dict(gt_bond=j,edge=key,reason=reason,paired_prediction=gt_to_pred.get(j)))
        # True anchors remove both atom detection and box-center displacement.
        anchors=np.array([x['center'] for x in ann['atoms']]);points=np.asarray([x['line'] for x in instances]).reshape(-1,2)
        idx=cKDTree(anchors).query(points)[1].reshape(-1,2) if len(points) else np.empty((0,2),int)
        pairs=[tuple(map(int,p)) if p[0]!=p[1] else None for p in idx]
        oracles['predicted_lines_gt_atom_anchors'].append(row_from_pairs(instances,pairs,truth))
        for tag,boxes,mp in [('gt_lines_predicted_boxes',[x['bbox'] for x in d['atoms']],mapping),('gt_lines_gt_boxes',[x['bbox'] for x in ann['atoms']],None)]:
            row,_=metric_row(gtinst,ann,[dict(bbox=x) for x in boxes],mp if mp is not None else {i:i for i in range(len(boxes))},'forward',0.)
            oracles[tag].append(row)
        # Counterfactuals conditioned on the label-free region assignment.
        for tag,change_line,change_class in [('region_matched_gt_lines',True,False),('region_matched_gt_classes',False,True),('region_matched_gt_lines_classes',True,True)]:
            changed=[dict(x) for x in instances]
            for i,j in match.items():
                if change_line:changed[i]['line']=gtl[j]
                if change_class:changed[i]['label']=band['labels'][j]
            row,_=metric_row(changed,ann,d['atoms'],mapping,'forward',0.);oracles[tag].append(row)
        # Deployment-compatible graph dedup: one highest-score bond per atom pair.
        keep=[];seen=set()
        for i in sorted(linked,key=lambda i:-instances[i]['score']):
            e=linked[i];pair=tuple(sorted((e['begin'],e['end'])))
            if pair not in seen:seen.add(pair);keep.append(i)
        keep+=d['unlinked'];changed=[instances[i] for i in sorted(keep)]
        row,_=metric_row(changed,ann,d['atoms'],mapping,'forward',0.);oracles['one_bond_per_predicted_atom_pair'].append(row)
        stats['duplicate_atom_pair_candidates']+=len(instances)-len(changed)
        # Pure assembled topology, without treating rejected candidates as edges.
        stats['assembled_typed_edges_exact_ignoring_unlinked']+=int(pred==truth)
        for threshold in [.2,.3,.4,.5,.6,.7,.8]:
            selected=[x for x in instances if x['score']>=threshold]
            row,_=metric_row(selected,ann,d['atoms'],mapping,'forward',0.);thresholds[str(threshold)].append(row)
        samples.append(dict(sample_id=d['sample_id'],image=d['record']['image'],atoms=n,bonds=b,atom_exact=d['components']['atom_exact'],
            bond_exact=bool(original['exact']),fp=dict(counts),missing=missitems,extra=items,geometry_matches=len(match),
            original=original,post_dedup=oracles['one_bond_per_predicted_atom_pair'][-1]))
    summary=lambda v:dict(n=len(v),median=float(np.median(v)) if v else None,p90=float(np.quantile(v,.9)) if v else None,p10=float(np.quantile(v,.1)) if v else None)
    report=dict(counts=dict(stats),fp_categories=dict(fp),fn_diagnostic_categories=dict(fn),
        by_complexity={k:aggregate(v) for k,v in buckets.items()},
        by_class={NAMES[k]:dict(gt=class_gt[k],pred=class_pred[k],tp=class_tp[k],precision=class_tp[k]/max(1,class_pred[k]),recall=class_tp[k]/max(1,class_gt[k]),f1=2*class_tp[k]/max(1,class_pred[k]+class_gt[k])) for k in range(6)},
        geometry={k:summary(v) for k,v in geometry.items()},vertical={k:dict(v) for k,v in vertical.items()},
        geometry_class_confusion={NAMES[t]+' -> '+NAMES[p]:n for (t,p),n in confusion.items() if t!=p},
        oracles={k:aggregate(v) for k,v in oracles.items()},thresholds={k:aggregate(v) for k,v in thresholds.items()},
        diagnostic_matching='Label-free region Hungarian matching: IoU >=0.1 or tight center+endpoint geometry; these pair-based attributions/counterfactuals are diagnostics, not the official exact score.',
        no_production_changes=True)
    assert sum(fp.values())==804 and sum(fn.values())==410
    write(a.output/'report.json',report);write(a.output/'samples.json',samples);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
