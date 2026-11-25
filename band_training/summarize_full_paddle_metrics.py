"""Train/validation evaluation with one frozen checkpoint and Paddle en consensus."""
import argparse,csv,gzip,hashlib,json,shutil,zipfile
from pathlib import Path
from collections import Counter
from .continue_indigo import match_boxes
from .geometry import edge_key,target_edges
from .mixed_data import ATOM_CLASSES

CHARGE_NAMES=['-4','-2','-1','+1','+2']
BOND_NAMES=['single','double','triple','aromatic','wedge','dash','type_pending']


def rates(tp,pred,gt):
    return dict(tp=tp,predicted=pred,ground_truth=gt,fp=pred-tp,fn=gt-tp,
        precision=tp/pred if pred else None,recall=tp/gt if gt else None,f1=2*tp/(pred+gt) if pred+gt else None)


def summarize_split(root):
    read=lambda p:json.loads(p.read_text())
    source=read(root/'prediction_cache/report.json');counts=source['counts'];n=counts['images']
    refs=read(root/'reference_verified/report.json');assert refs['status']=='passed',refs['errors'][:2]
    graph=read(root/'graph_ocr/functional_report.json');ocr=read(root/'graph_ocr/report.json')
    assert ocr['ocr']['policy']=='en_consensus' and ocr['ocr']['languages']==['en']
    assert ocr['counts']['legacy_full_exact']==counts['legacy_full_exact']
    per={name:dict(tp=Counter(),pred=Counter(),gt=Counter()) for name in ['atom','charge','charge_owner','bond']};strict=Counter();shape=Counter();sample_rows=[]
    raw_graph={r['sample_id']:r for r in read(root/'graph_ocr/functional_samples.json')}
    references={r['sample_id']:r for r in read(root/'reference_verified/samples.json')}
    with gzip.open(root/'prediction_cache/predictions.jsonl.gz','rt') as f:
        for d in map(json.loads,f):
            for name,key in [('atom','atoms'),('charge','charges')]:
                truth=d['ann'][key];pred=d[key]
                mapping=d['atom_mapping'] if name=='atom' else d['charge_detection_mapping']
                per[name]['tp'].update(pred[int(i)]['label'] for i in mapping)
                per[name]['pred'].update(r['label'] for r in pred);per[name]['gt'].update(r['category_id'] for r in truth)
                strict.update({name+'_tp':len(match_boxes(pred,truth,min_iou=.5)),name+'_pred':len(pred),name+'_gt':len(truth)})
            mapping={int(k):v for k,v in d['atom_mapping'].items()};gt=target_edges(d['ann']);actual=Counter()
            expected_charge=Counter((q['atom_index'],q['category_id']) for q in d['ann']['charges'])
            actual_charge=Counter((mapping.get(a['atom_index'],-1),q['label']) for q,a in zip(d['charges'],d['charge_assignments']))
            per['charge_owner']['tp'].update({label:sum(v for key,v in (expected_charge&actual_charge).items() if key[1]==label) for label in range(13,18)})
            per['charge_owner']['pred'].update(q['label'] for q in d['charges']);per['charge_owner']['gt'].update(q['category_id'] for q in d['ann']['charges'])
            for e in d['bonds']:
                u,v=mapping.get(e['begin'],-1),mapping.get(e['end'],-1)
                if min(u,v)>=0:actual[edge_key(u,v,e['label'])]+=1
            per['bond']['tp'].update({label:sum(v for e,v in (gt&actual).items() if e[2]==label) for label in range(6)})
            per['bond']['pred'].update(r['label'] for r in d['instances'])
            per['bond']['pred'][6]+=len(d.get('untyped_lines',[]))
            per['bond']['gt'].update({label:sum(v for e,v in gt.items() if e[2]==label) for label in range(6)})
            shape.update(images=1,atoms=len(d['ann']['atoms']),bonds=len(d['ann']['bonds']),charges=len(d['ann']['charges']))
            sample_rows.append(dict(sample_id=d['sample_id'],image=d['record']['image'],annotation=d['record']['annotation'],atoms=len(d['ann']['atoms']),bonds=len(d['ann']['bonds']),charges=len(d['ann']['charges']),
                **d['components'],graph_star_exact=raw_graph[d['sample_id']]['graph_star_exact'],graph_ocr_exact=raw_graph[d['sample_id']]['graph_ocr_exact']))
    objects={};class_rows=[]
    for name,p in per.items():
        objects[name]=rates(sum(p['tp'].values()),sum(p['pred'].values()),sum(p['gt'].values()))
        if name=='bond':assert objects[name]['fp']==counts['bond_fp'] and objects[name]['fn']==counts['bond_fn']
        labels=list(enumerate(ATOM_CLASSES)) if name=='atom' else list(enumerate(CHARGE_NAMES,13)) if name in ['charge','charge_owner'] else list(enumerate(BOND_NAMES))
        for label,title in labels:class_rows.append(dict(component=name,label=label,name=title,**rates(p['tp'][label],p['pred'][label],p['gt'][label])))
    for name in ['atom','charge']:
        assert objects[name]['tp']==counts[name+'_tp'] and objects[name]['predicted']==counts[name+'_pred'] and objects[name]['ground_truth']==counts[name+'_gt']
        objects[name+'_iou050']=rates(strict[name+'_tp'],strict[name+'_pred'],strict[name+'_gt'])
    whole={k:dict(correct=counts[k],total=n,rate=counts[k]/n) for k in ['atom_exact','charge_detection_exact','charge_owner_exact','bond_exact','legacy_full_exact']}
    for key in ['graph_star_exact','graph_ocr_exact']:
        c=graph['groups']['all'][key];whole[key]=dict(correct=c,total=n,rate=c/n)
    charge_subsets=dict(charged_owner=dict(correct=counts['charged_owner_exact'],total=counts['charged_images']),
        charged_detection=dict(correct=counts['charged_detection_exact'],total=counts['charged_images']),
        neutral_no_false_positive=dict(correct=counts['neutral_charge_exact'],total=counts['neutral_images']))
    alias_keys={(sid,r['atom_id']) for sid,s in references.items() for r in s['dummy_labels'] if r['kind']=='alias_or_rsite'}
    crops=read(root/'graph_ocr/ocr_crops.json')
    matched=[r for r in crops if r['kind']=='predicted_crop' and (r['sample_id'],r['reference_atom']) in alias_keys]
    oracle=[r for r in crops if r['kind']=='gt_crop_diagnostic' and (r['sample_id'],r['reference_atom']) in alias_keys]
    ocr_labels=dict(predicted_matched_alias=dict(correct=sum(r['exact'] for r in matched),total=len(matched)),
        gt_alias_diagnostic=dict(correct=sum(r['exact'] for r in oracle),total=len(oracle)),
        alias_images=graph['groups'].get('with_alias',{}),unsupported_elements=graph['groups'].get('with_unsupported_element',{}))
    result=dict(images=n,source=source,counts=counts,objects=objects,whole_images=whole,charge_subsets=charge_subsets,
        per_class=class_rows,ocr_labels=ocr_labels,reference_audit=refs,ocr_backend=ocr['ocr'],dataset_shape=dict(shape),
        ocr_lost_images=whole['graph_star_exact']['correct']-whole['graph_ocr_exact']['correct'],
        strict_component_combinations={k.removeprefix('components_'):v for k,v in ocr['counts'].items() if k.startswith('components_')})
    return result,sample_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();splits={};samples={}
    for split in ['train','val']:splits[split],samples[split]=summarize_split(a.root/split)
    assert splits['train']['source']['checkpoint_sha256']==splits['val']['source']['checkpoint_sha256']
    for key in ['backend','versions','model_sha256','scales','margins','languages','policy']:
        assert splits['train']['ocr_backend'][key]==splits['val']['ocr_backend'][key],('OCR settings differ',key)
    manifests={s:json.loads((a.root/s/'prediction_cache/evaluated_manifest.json').read_text()) for s in splits}
    assert not ({r['sample_id'] for r in manifests['train']}&{r['sample_id'] for r in manifests['val']})
    group_overlap={r['group'] for r in manifests['train']}&{r['group'] for r in manifests['val']};assert not group_overlap
    for split,rows in manifests.items():assert len({r['image'] for r in rows})==len(rows),'Repeated image file in '+split
    assert not ({r['image'] for r in manifests['train']}&{r['image'] for r in manifests['val']})
    report=dict(status='completed',schema='full_dataset_paddle_en_consensus_v1',splits=splits,
        total_unique_images=sum(v['images'] for v in splits.values()),train_val_id_overlap=0,train_val_group_overlap=len(group_overlap),
        evaluation_rules=dict(inference='Image pixels only; predicted atom/charge boxes, line and band heads. No GT identities/anchors/labels used in inference or assembly.',
            atom='Existing 13-class label-aware Hungarian matching; IoU>=0.1 OR center distance<max(8 px,0.5 GT box diagonal), in 640x640 letterbox coordinates. Other is one detector class.',
            charge='Existing five-class label-aware Hungarian box matching, IoU>=0.25. Ownership correctness additionally checks formal charge value and mapped owner.',
            supplemental_strict_boxes='Separate atom/charge label-aware Hungarian box matching at IoU>=0.50, with no center-distance fallback.',
            bond='All connections, six bond classes and wedge/dash endpoint direction checked against GT after geometry-only assembly. Wrong class or direction yields FP+FN. Pending/unlinked candidates count as FP.',
            strict_graph='atom_exact AND charge_owner_exact AND bond_exact; no extra/missing detection allowed.',
            graph_star='Full property graph isomorphism: functional-group/alias/R-site labels become anonymous dummy atoms, true rare elements retain identity. Aromatic/Kekule orders independently normalized.',
            graph_ocr='Same full graph plus literal PaddleOCR English PP-OCRv4 labels, scales 1/2/3/4, margins 1/3, majority normalized text and confidence tie-break. No GT vocabulary or correction.',
            not_scored='Isotope/implicit-H/atom-CIP/E-Z are not independently scored. MCS partial matches never count as correct.',
            split_scope='Train: each image evaluated once, no color jitter and no sampling repeats. Training accuracy is in-sample; validation is development-set performance, not a newly held-out test.'))
    out=a.root/'review';out.mkdir(exist_ok=True);(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    for split,rows in samples.items():
        with (out/(split+'_samples.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with (out/(split+'_per_class.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(splits[split]['per_class'][0]));w.writeheader();w.writerows(splits[split]['per_class'])
    metrics=[]
    for split,r in splits.items():
        for k,v in r['objects'].items():metrics.append(dict(split=split,metric=k,**v))
    with (out/'object_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    whole=[dict(split=s,metric=k,**v) for s,r in splits.items() for k,v in r['whole_images'].items()]
    with (out/'whole_image_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(whole[0]));w.writeheader();w.writerows(whole)
    lines=['全量原训练数据评估：只使用PaddleOCR英文四尺度+投票。',
        '训练重复采样去重；train/val图片ID和分子group均无交叉；推理不使用GT boxes或端点。','']
    for split,r in splits.items():
        lines.append(f"{split}: {r['images']}张图；checkpoint epoch {r['source']['checkpoint_epoch']}；SHA256 {r['source']['checkpoint_sha256']}")
        for k,v in r['objects'].items():
            fmt=lambda x:'N/A' if x is None else f'{100*x:.4f}%'
            lines.append(f"  {k}: P={fmt(v['precision'])}, R={fmt(v['recall'])}, F1={fmt(v['f1'])}; TP={v['tp']}, FP={v['fp']}, FN={v['fn']}")
        for k,v in r['whole_images'].items():lines.append(f"  {k}: {v['correct']}/{v['total']} = {100*v['rate']:.4f}%")
        for k,v in r['charge_subsets'].items():lines.append(f"  {k}: {v['correct']}/{v['total']}")
        lines.append('')
    lines+=['评估口径详见report.json；训练集成绩不代表泛化成绩。']
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    code=out/'source';code.mkdir(exist_ok=True)
    for name in ['summarize_full_paddle_metrics.py','cache_final_predictions.py','audit_graph_references.py','evaluate_graph_ocr.py',
                 'scored_bond_postprocess.py','scored_bond_model.py','paddle_ocr_replay.py','paddle_multiscale_worker.py','prepare_paddle_ocr_requests.py',
                 'molecular_graph_metrics.py','continue_indigo.py','geometry.py','charge_association.py','charge_adaptive.py','mixed_data.py','rescore_functional_groups.py']:
        shutil.copyfile(Path(__file__).with_name(name),code/name)
    with zipfile.ZipFile(a.root/'review.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(a.root))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
