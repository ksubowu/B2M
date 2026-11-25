"""Attribute frozen atom FNs to score/argmax/NMS/matching stages and render examples."""
import argparse,csv,gzip,json,math,shutil,zipfile
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.optimize import linear_sum_assignment
from .continue_indigo import iou_matrix
from .mixed_data import ATOM_CLASSES,letterbox_transform

NAMES=ATOM_CLASSES+['charge-4','charge-2','charge-1','charge+1','charge+2']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'review';out.mkdir(exist_ok=True)
    geometry=json.loads((a.root/'geometry/report.json').read_text());details={}
    for split in ['train','val']:
        for row in json.loads((a.root/'geometry'/(split+'_samples.json')).read_text()):details[split,row['sample_id']]=row
    stages=defaultdict(Counter);conflicts=defaultdict(Counter);rows=[];examples={};allraw={};low_score=defaultdict(Counter);matching=defaultdict(Counter)
    with gzip.open(a.root/'raw_queries/queries.jsonl.gz','rt') as f:
        for d in map(json.loads,f):
            split=d['split'];info=details[split,d['sample_id']];allraw[split,d['sample_id']]=d
            boxes=np.asarray(d['boxes']);scores=np.asarray(d['class_scores']);top=np.asarray(d['top_class']);topscore=np.asarray(d['top_score'])
            kept=np.asarray(d['kept_atom_queries'],int);kept_set=set(kept.tolist());pc=boxes[:,:2]+boxes[:,2:]/2
            pred=d['cached_atoms'];truth=d['ann_atoms'];additional=0;newtp=0
            if pred and truth:
                pb=np.asarray([p['bbox'] for p in pred]);gb=np.asarray([g['bbox'] for g in truth]);cross=iou_matrix(pb,gb)
                dist=np.linalg.norm((pb[:,:2]+pb[:,2:]/2)[:,None]-(gb[:,:2]+gb[:,2:]/2)[None],axis=-1);diag=np.linalg.norm(gb[:,2:],axis=1)
                same=np.array([p['label'] for p in pred])[:,None]==np.array([g['category_id'] for g in truth])[None]
                valid=same&((cross>=.1)|(dist<np.maximum(8.,diag*.5)[None]))
                cost=3*(1-cross)+dist/(4+diag[None]/2);ii,jj=linear_sum_assignment(np.where(valid,cost,1e9));newtp=int(valid[ii,jj].sum())
            assert newtp>=len(d['atom_mapping'])
            additional=newtp-len(d['atom_mapping']);matching[split].update(additional_tp=additional,images_with_more_tp=int(additional>0),
                newly_atom_exact_images=int(newtp==len(pred)==len(truth)))
            for miss in info['missing']:
                g=np.asarray(miss['bbox']);label=miss['label'];gc=g[:2]+g[2:]/2;diag=np.linalg.norm(g[2:]);distance=np.linalg.norm(pc-gc,axis=1)
                iou=iou_matrix(boxes,[g]).ravel();gate=(iou>=.1)|(distance<max(8.,diag*.5))
                correct=np.flatnonzero(gate&(top==label)&(topscore>.25));retained=[q for q in correct if q in kept_set]
                strong=np.flatnonzero(gate&(scores[:,label]>.25));localized=np.flatnonzero(gate)
                best=int(localized[np.argmax(scores[localized,label])]) if len(localized) else int(np.argmax(scores[:,label]))
                pair=miss['paired_prediction'];suppression=None
                if miss['reason']=='residual_same_class_assignment':reason='retained_same_class_unused_by_matching'
                elif retained:reason='retained_same_class_already_used_for_neighbor'
                elif len(correct):
                    reason='correct_class_candidate_suppressed_by_NMS';best=int(correct[np.argmax(topscore[correct])])
                    unclipped=np.asarray(d['xyxy_unclipped']);rawxywh=np.c_[unclipped[:,:2],unclipped[:,2:]-unclipped[:,:2]]
                    overlap=iou_matrix(rawxywh[[best]],rawxywh[kept]).ravel();eligible=np.flatnonzero((overlap>.4)&(topscore[kept]>=topscore[best]))
                    if len(eligible):
                        j=int(eligible[np.argmax(topscore[kept[eligible]])]);q=int(kept[j]);suppression=dict(query=q,label=int(top[q]),category=NAMES[top[q]],score=float(topscore[q]),iou=float(overlap[j]),box=boxes[q].tolist())
                        conflicts[split][f"{ATOM_CLASSES[label]}<-{NAMES[top[q]]}"]+=1
                elif pair is not None:reason='nearby_retained_prediction_has_wrong_class'
                elif len(strong):reason='expected_class_loses_query_argmax'
                else:
                    unmatched=[x['prediction'] for x in info['false_positives'] if x['label']==label]
                    nearby=[i for i in unmatched if np.linalg.norm(np.array(d['cached_atoms'][i]['bbox'][:2])+np.array(d['cached_atoms'][i]['bbox'][2:])/2-gc)<max(16.,diag)]
                    if nearby:reason='nearby_same_class_box_fails_geometry_gate'
                    elif not len(localized):reason='no_raw_box_localized_at_GT'
                    else:reason='expected_class_score_below_threshold'
                stages[split][reason]+=1
                if reason=='expected_class_score_below_threshold':
                    v=float(scores[best,label]);low_score[split]['0.20-0.25' if v>.2 else '0.10-0.20' if v>.1 else '<=0.10']+=1
                record=dict(split=split,sample_id=d['sample_id'],atom=miss['atom'],category=ATOM_CLASSES[label],symbol=miss['symbol'],stage=reason,
                    visible_label=miss['visible_label'],bbox=miss['bbox'],bbox_kind=miss['bbox_kind'],nearest_gt_distance=miss['nearest_gt_distance'],gt_box_overlap=miss['gt_box_overlap'],
                    expected_class_score=float(scores[best,label]),candidate_query=best,candidate_box=boxes[best].tolist(),
                    candidate_top_class=NAMES[top[best]],candidate_top_score=float(topscore[best]),candidate_iou=float(iou[best]),
                    candidate_center_distance=float(distance[best]),suppression=suppression,geometry_pair=pair)
                rows.append(record);examples.setdefault((split,reason),record)
    for split,expected in [('train',714),('val',128)]:assert sum(stages[split].values())==expected
    report=dict(status='completed',geometry=geometry,raw_capture=json.loads((a.root/'raw_queries/report.json').read_text()),
        stages={k:dict(v) for k,v in stages.items()},nms_suppressor_classes={k:dict(v) for k,v in conflicts.items()},low_score_bins={k:dict(v) for k,v in low_score.items()},
        interpretation='FN is a scoring result, not synonymous with no model output. Stage attribution is ordered: unused same-class match; neighboring assignment; NMS; retained wrong class; query argmax; nearby localization; low score. Geometry uses the original metric gate. This is an operational diagnosis, not proof of a unique visual cause; no evaluation metrics or model weights changed.')
    if (out/'structure_strata.json').exists():report['structure_strata']=json.loads((out/'structure_strata.json').read_text())
    report['matching_counterfactual']={}
    for split in ['train','val']:
        old=geometry['splits'][split]['totals'];m=matching[split];newtp=old['atom_tp']+m['additional_tp'];pred=old['atom_tp']+old['fp'];gt=old['atom_gt']
        report['matching_counterfactual'][split]=dict(m,old_fn=old['fn'],new_fn=gt-newtp,old_fp=old['fp'],new_fp=pred-newtp,
            old_atom_f1=2*old['atom_tp']/(pred+gt),new_atom_f1=2*newtp/(pred+gt),
            note='Diagnostic only: same class and geometry gates, maximize accepted one-to-one match count before minimizing cost. Model predictions unchanged; historical metrics not overwritten. Does not change coordinate-independent graph_ocr isomorphism.')
    report['worst_images']={}
    for split in ['train','val']:
        ranked=sorted([r for r in details.values() if r['split']==split],key=lambda r:r['fn'],reverse=True)
        report['worst_images'][split]=dict(top5=[dict(sample_id=r['sample_id'],fn=r['fn'],fp=r['fp'],gt_atoms=r['gt_atoms']) for r in ranked[:5]],
            top5_fn=sum(r['fn'] for r in ranked[:5]))
    (out/'report.json').write_text(json.dumps(report,indent=2));(out/'missing_atoms.json').write_text(json.dumps(rows,indent=2))
    fields=['split','sample_id','atom','category','symbol','stage','visible_label','bbox_kind','nearest_gt_distance','gt_box_overlap','expected_class_score','candidate_top_class','candidate_top_score','candidate_iou','candidate_center_distance']
    with (out/'missing_atoms.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    class_rows=[]
    for name in ATOM_CLASSES:
        t=geometry['splits']['train']['classes'][name];v=geometry['splits']['val']['classes'][name]
        class_rows.append(dict(category=name,train_gt=t['gt'],train_fn=t.get('fn',0),train_fn_rate=t['fn_rate'],train_fp=t.get('fp',0),val_gt=v['gt'],val_fn=v.get('fn',0),val_fn_rate=v['fn_rate'],val_fp=v.get('fp',0)))
    with (out/'classes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(class_rows[0]));w.writeheader();w.writerows(class_rows)
    images=out/'examples';images.mkdir(exist_ok=True)
    for old in images.glob('[0-9][0-9]_*.png'):old.unlink()
    dataset=Path('/mnt/hdd2/bo/work/cvs/chemcv_data/real/indigo_full_v3_20261116')
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(font_path,14)
    selected=list(examples.values())
    # Add pure-FP examples: atom-error image counts are not FN-image counts.
    for split in ['train','val']:
        pure=[r for r in details.values() if r['split']==split and r['fp'] and not r['fn']]
        if pure:
            r=max(pure,key=lambda r:r['fp']);selected.append(dict(split=split,sample_id=r['sample_id'],stage='only_false_positives',atom=None))
        worst=max([r for r in details.values() if r['split']==split],key=lambda r:r['fn'])
        base=next(r for r in rows if r['split']==split and r['sample_id']==worst['sample_id'])
        selected.append(dict(base,stage='worst_FN_image'))
    index=[]
    for i,r in enumerate(selected):
        info=details[r['split'],r['sample_id']];raw=allraw[r['split'],r['sample_id']]
        with Image.open(dataset/info['record']['image']) as im:original=im.convert('RGB')
        scale,offset,shape=letterbox_transform(*original.size);base=Image.new('RGB',(640,640),'white');base.paste(original.resize(shape,Image.Resampling.LANCZOS),tuple(offset))
        panels=[base,base.copy(),base.copy()]
        missing={x['atom'] for x in info['missing']};extra={x['prediction'] for x in info['false_positives']}
        for j,t in enumerate(raw['ann_atoms']):
            x,y,w,h=t['bbox'];dr=ImageDraw.Draw(panels[1]);color='red' if j in missing else '#4e79a7'
            dr.rectangle((x,y,x+w,y+h),outline=color,width=3 if j in missing and r['stage']!='worst_FN_image' else 1)
            if j in missing and r['stage']!='worst_FN_image' and (len(missing)<=5 or j==r.get('atom')):dr.text((x,y-15),f"{j}:{t['symbol']}",fill=color,font=font)
        for j,t in enumerate(raw['cached_atoms']):
            x,y,w,h=t['bbox'];dr=ImageDraw.Draw(panels[2]);color='#e67e00' if j in extra else '#1b8e3e'
            dr.rectangle((x,y,x+w,y+h),outline=color,width=2 if j in extra else 1)
            if j in extra:dr.text((x,y-15),f"{ATOM_CLASSES[t['label']]} {t['score']:.2f}",fill=color,font=font)
        if r.get('bbox'):
            x,y,w,h=r['bbox'];ImageDraw.Draw(panels[2]).rectangle((x,y,x+w,y+h),outline='red',width=3)
        canvas=Image.new('RGB',(1920,715),'white');dr=ImageDraw.Draw(canvas)
        for k,(panel,title) in enumerate(zip(panels,['Original','GT boxes: red = unmatched GT','Predictions: orange = FP; red = selected FN'])):
            canvas.paste(panel,(640*k,28));dr.text((640*k+5,5),title,fill='black',font=font)
        text=f"{r['split']} | {r['sample_id']} | {r['stage']}"
        dr.text((5,674),text,fill='black',font=font)
        if 'expected_class_score' in r:
            dr.text((5,694),f"GT={r['category']} ({r['symbol']}); raw expected score={r['expected_class_score']:.3f}; raw top={r['candidate_top_class']} {r['candidate_top_score']:.3f}",fill='black',font=font)
        name=f"{i:02d}_{r['split']}_{r['stage']}.png";canvas.save(images/name);index.append(dict(file=name,**r))
        if r['stage']=='worst_FN_image':
            b=np.array([a['bbox'] for a in raw['ann_atoms']]);lo=np.floor(b[:,:2].min(0)-8).clip(0,640);hi=np.ceil((b[:,:2]+b[:,2:]).max(0)+8).clip(0,640)
            bounds=tuple(map(int,[*lo,*hi]));zoom=Image.new('RGB',(1920,690),'white');zd=ImageDraw.Draw(zoom)
            for k,panel in enumerate(panels):
                crop=panel.crop(bounds);factor=min(630/crop.width,630/crop.height)
                crop=crop.resize((round(crop.width*factor),round(crop.height*factor)),Image.Resampling.LANCZOS)
                zoom.paste(crop,(640*k+(640-crop.width)//2,30+(630-crop.height)//2))
                zd.text((640*k+5,5),['Zoom original','Zoom GT boxes','Zoom actual predictions'][k],fill='black',font=font)
            zd.text((5,665),f"{r['split']} | GT molecule span {hi[0]-lo[0]-16:.0f} x {hi[1]-lo[1]-16:.0f}px in 640x640 input | FN={info['fn']} FP={info['fp']}",fill='black',font=font)
            zoom.save(images/(name.removesuffix('.png')+'_zoom.png'))
    (out/'examples.json').write_text(json.dumps(index,indent=2))
    lines=['Atom-error diagnosis. Counts refer to the same frozen evaluation.','']
    for split in ['train','val']:
        lines+=[split+': '+str(geometry['splits'][split]['totals']),str(dict(stages[split])),'']
    lines.append(report['interpretation']);(out/'report.txt').write_text('\n'.join(lines))
    source=out/'source';source.mkdir(exist_ok=True)
    for name in ['diagnose_atom_errors.py','capture_atom_queries.py','finalize_atom_diagnosis.py','atom_structure_strata.py']:
        shutil.copyfile(Path(__file__).with_name(name),source/name)
    with zipfile.ZipFile(a.root/'review.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(a.root))
    print(json.dumps(dict(stages={k:dict(v) for k,v in stages.items()},nms={k:dict(v) for k,v in conflicts.items()},low_score={k:dict(v) for k,v in low_score.items()}),indent=2),flush=True)


if __name__=='__main__':main()
