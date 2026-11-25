"""Reproducible pilot with independent box/band branches and held-out graph checks."""
import argparse
import copy
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.ops import nms,box_convert

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'repo'),str(ROOT/'repo/i2bl_webapp')]
from pair_training.train import load_model
from engine.core import YAMLConfig
from .model import BandDecoder,targets_for,band_loss,decode
from .geometry import aggregate,edge_metrics,atom_mapping,suppress,polygon_iou,link


def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False))


def decode_atoms(out,threshold=.25):
    scores,labels=out['pred_logits'][0].sigmoid().max(-1)
    boxes=box_convert(out['pred_boxes'][0], 'cxcywh','xyxy')*640
    result=[]
    for charge in [False,True]:
        keep=torch.where((scores>threshold)&((labels>=13) if charge else (labels<13)))[0]
        keep=keep[nms(boxes[keep],scores[keep],.4)]
        for i in keep.tolist():
            xyxy=boxes[i].detach().float().cpu().numpy().clip(0,640)
            result.append({'label':int(labels[i]),'score':float(scores[i]),
                           'bbox':[float(xyxy[0]),float(xyxy[1]),float(xyxy[2]-xyxy[0]),float(xyxy[3]-xyxy[1])]})
    return result


def charge_exact(predictions,atoms,mapping,ann,assignments=None):
    expected=Counter((q['atom_index'],q['category_id']) for q in ann['charges'])
    actual=Counter()
    if assignments is not None:
        charges=[q for q in predictions if q['label']>=13]
        assert len(charges)==len(assignments)
        for q,assignment in zip(charges,assignments):
            owner=assignment['atom_index']
            actual[(mapping.get(owner,-1) if owner is not None else -1,q['label'])]+=1
        return int(expected==actual)
    for q in predictions:
        if q['label']<13:continue
        x,y,w,h=q['bbox'];center=np.array([x+w/2,y+h/2])
        scores=[]
        for i,a in enumerate(atoms):
            ax,ay,aw,ah=a['bbox']
            dist=np.linalg.norm(np.maximum(np.maximum([ax,ay]-center,center-[ax+aw,ay+ah]),0))
            scores.append((dist+.02*np.linalg.norm(center-[ax+aw/2,ay+ah/2]),i))
        if not scores:actual[(-1,q['label'])]+=1;continue
        dist,i=min(scores);actual[(mapping.get(i,-1),q['label'])]+=1
    return int(expected==actual)


def make_cache(dataset,base,path,device):
    manifest=json.loads((dataset/'manifest.json').read_text());cached=[]
    for idx,row in enumerate(manifest):
        ann=json.loads((dataset/row['truth']).read_text())
        target=json.loads((dataset/row['band_target']).read_text())
        with Image.open(dataset/row['image']) as im:image=TF.to_tensor(im.convert('RGB')).unsqueeze(0).to(device)
        normalized=TF.normalize(image,[.485,.456,.406],[.229,.224,.225])
        with torch.no_grad():
            features=base.encoder(base.backbone(normalized))
            if isinstance(features,tuple):features=features[0]
            out=base.box_decoder(features)
        gray=image.mean(1,keepdim=True)
        atom_boxes=[a['bbox'] for a in ann['atoms']]+[q['bbox'] for q in ann['charges']]
        labels=[a['category_id'] for a in ann['atoms']]+[q['category_id'] for q in ann['charges']]
        xywh=torch.tensor(atom_boxes,dtype=torch.float32)
        cxcywh=xywh.clone();cxcywh[:,:2]+=xywh[:,2:]/2;cxcywh/=640
        cached.append(dict(row=row,ann=ann,band=target,features=[f.detach().cpu().half()[0] for f in features],
                           gray=gray.detach().cpu().half()[0],baseline_atoms=decode_atoms(out),
                           atom_target={'labels':torch.tensor(labels),'boxes':cxcywh}))
        if (idx+1)%32==0:print('Cached',idx+1,'/',len(manifest),flush=True)
    torch.save(cached,path)
    return cached


def to_device(records,device):
    for r in records:
        r['features']=[v.to(device).float() for v in r['features']]
        r['gray']=r['gray'].to(device).float()
        heat,entries=targets_for(r['band']['labels'],r['band']['quads'],'cpu')
        r['target']=(heat.to(device),[(l,y,x,o.to(device),q.to(device)) for l,y,x,o,q in entries])
        r['atom_target']={k:v.to(device) for k,v in r['atom_target'].items()}
    return records


def batch(records,indices):
    f=[torch.stack([records[i]['features'][j] for i in indices]) for j in range(3)]
    g=torch.stack([records[i]['gray'] for i in indices])
    return f,g,[records[i]['target'] for i in indices]


def predictions(model,records,indices):
    model.eval();result={}
    with torch.no_grad():
        for i in indices:
            r=records[i]
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model([v[None] for v in r['features']],r['gray'][None])
            result[i]=decode(out,minimum=.03)
    return result


def evaluate(records,indices,preds,threshold,atom_key='baseline_atoms'):
    oracle=[];full=[];atom_tp=atom_pred=atom_gt=0;strict=0;charge_ok=0;region_tp=region_gt=region_pred=0
    per_type={k:[0,0,0] for k in range(6)}
    for i in indices:
        r=records[i];ann=r['ann']
        ps=suppress([p for p in preds[i] if p['score']>=threshold])
        oracle.append(edge_metrics(ps,ann,[a['bbox'] for a in ann['atoms']]))
        ap=r[atom_key];atoms=[a for a in ap if a['label']<13]
        mapping,tp=atom_mapping(atoms,ann)
        atom_tp+=tp;atom_pred+=len(atoms);atom_gt+=len(ann['atoms'])
        row=edge_metrics(ps,ann,[a['bbox'] for a in atoms],mapping)
        full.append(row)
        charge=charge_exact(ap,atoms,mapping,ann);charge_ok+=charge
        strict+=int(row['exact'] and tp==len(atoms)==len(ann['atoms']) and charge)
        # Region detection uses a 0.25 polygon IoU threshold and correct type.
        gts=list(zip(r['band']['labels'],r['band']['quads']));used=set();hits=0
        for p in ps:
            matches=[(polygon_iou(p['quad'],q),j) for j,(label,q) in enumerate(gts) if j not in used and label==p['label']]
            best=max(matches,default=(0,-1))
            hit=best[0]>=.25
            if hit:used.add(best[1]);hits+=1
            per_type[p['label']][0]+=int(hit);per_type[p['label']][1]+=1
        for label,q in gts:per_type[label][2]+=1
        region_tp+=hits;region_pred+=len(ps);region_gt+=len(gts)
    return dict(oracle_atoms=aggregate(oracle),predicted_atoms=aggregate(full),
                atom_precision=atom_tp/max(1,atom_pred),atom_recall=atom_tp/max(1,atom_gt),
                charge_graph_exact=charge_ok/max(1,len(indices)),full_graph_exact=strict/max(1,len(indices)),
                region_precision_iou025=region_tp/max(1,region_pred),region_recall_iou025=region_tp/max(1,region_gt),
                region_per_type={str(k):dict(tp=v[0],pred=v[1],gt=v[2]) for k,v in per_type.items()})


def ceiling(records,indices):
    rows=[];details=[]
    for i in indices:
        r=records[i]
        ps=[dict(label=l,quad=np.asarray(q),score=1.) for l,q in zip(r['band']['labels'],r['band']['quads'])]
        metric=edge_metrics(ps,r['ann'],[a['bbox'] for a in r['ann']['atoms']]);rows.append(metric)
        if not metric['exact']:details.append(dict(image=r['row']['image'],**metric))
    return dict(**aggregate(rows),failures=details)


def train_band(model,records,train,steps,batch_size,lr,log,prefix,seed,eval_every=0,cal=None,save=None):
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    rng=random.Random(seed);start=time.time();history=[];best=-1
    for step in range(steps):
        model.train();ids=rng.sample(train,min(batch_size,len(train)))
        features,gray,targets=batch(records,ids)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(features,gray)
        losses=band_loss({k:v.float() for k,v in out.items()},targets)
        if not torch.isfinite(losses['total']):raise RuntimeError('Nonfinite band loss')
        losses['total'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
        assert float(norm)>0,'Disconnected band gradients'
        optimizer.step()
        if step%25==0 or step==steps-1:
            row=dict(stage=prefix,step=step+1,seconds=round(time.time()-start,1),
                     **{k:float(v.detach()) for k,v in losses.items()})
            history.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
        if eval_every and ((step+1)%eval_every==0 or step==steps-1):
            preds=predictions(model,records,cal)
            scores=[]
            for threshold in [.10,.20,.30,.40,.50,.60,.70]:
                m=evaluate(records,cal,preds,threshold)
                scores.append((m['oracle_atoms']['f1'],threshold,m))
            score,threshold,metric=max(scores,key=lambda x:x[0])
            row=dict(stage=prefix,step=step+1,calibration_f1=score,threshold=threshold,
                     calibration_graph_exact=metric['oracle_atoms']['graph_exact'])
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if score>best:
                best=score
                if save:torch.save(dict(band=model.state_dict(),threshold=threshold,step=step+1,calibration=metric),save)
    return history


def train_atom(base,records,train,cal,steps,batch_size,log,output):
    cfg=YAMLConfig(str(ROOT/'config/train_mp123_full.yml'));criterion=cfg.criterion.box_criterion.to('cuda').train()
    decoder=base.box_decoder.requires_grad_(True)
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=2e-5,weight_decay=1e-4)
    rng=random.Random(20261114);start=time.time()
    for step in range(steps):
        decoder.train();ids=rng.sample(train,min(batch_size,len(train)))
        features,_,_=batch(records,ids);targets=[records[i]['atom_target'] for i in ids]
        optimizer.zero_grad(set_to_none=True)
        # FP32 preserves the reference deformable-sampling/criterion path.
        out=decoder(features,targets=targets)
        losses=criterion(out,targets);loss=sum(losses.values())
        assert torch.isfinite(loss),'Nonfinite atom loss'
        loss.backward();torch.nn.utils.clip_grad_norm_(decoder.parameters(),1,error_if_nonfinite=True);optimizer.step()
        if step%25==0 or step==steps-1:
            row=dict(stage='atom_finetune',step=step+1,loss=float(loss.detach()),seconds=round(time.time()-start,1))
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    decoder.eval()
    torch.save(decoder.state_dict(),output/'atom_decoder.pth')
    with torch.no_grad():
        for r in records:r['finetuned_atoms']=decode_atoms(decoder([v[None] for v in r['features']]))


def visualize(records,ids,preds,threshold,root,output):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for number,i in enumerate(ids[:6]):
        r=records[i];im=Image.open(root/r['row']['image']).convert('RGB');w,h=im.size
        panels=[im.copy(),im.copy(),im.copy()]
        gt=ImageDraw.Draw(panels[1]);pr=ImageDraw.Draw(panels[2])
        for label,q in zip(r['band']['labels'],r['band']['quads']):
            poly=[tuple(p) for p in q];gt.line(poly+[poly[0]],fill='green',width=2)
        for a in r.get('finetuned_atoms',r['baseline_atoms']):
            x,y,bw,bh=a['bbox'];pr.rectangle((x,y,x+bw,y+bh),outline='blue' if a['label']<13 else 'red',width=1)
        for p in suppress([p for p in preds[i] if p['score']>=threshold]):
            poly=[tuple(v) for v in p['quad']];pr.line(poly+[poly[0]],fill='orange',width=2)
            x,y=np.asarray(p['quad']).mean(0);pr.text((x,y),str(p['label']+1),font=font,fill='purple')
        canvas=Image.new('RGB',(w*3,h+40),'white');draw=ImageDraw.Draw(canvas)
        for j,(p,title) in enumerate(zip(panels,['Original test image','Ground-truth band regions','Predicted bands + atom/charge boxes'])):
            canvas.paste(p,(j*w,40));draw.text((j*w+8,8),title,font=font,fill='black')
        canvas.save(output/f'test_{number:02d}.png')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--tiny-steps',type=int,default=300);p.add_argument('--atom-steps',type=int,default=300)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=20261113)
    p.add_argument('--base',type=Path,default=ROOT/'runs/human_joint_full_20261027/best_base.pth')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    write(args.output/'config.json',{**vars(args),'dataset':str(args.dataset.resolve()),'output':str(args.output.resolve()),
                                   'base':str(args.base.resolve()),'seed':args.seed,'architecture':'dense_band_quad_v1',
                                   'frozen_backbone_encoder':True,'atom_association_loss':False,
                                   'band_losses':{'exist_type':'center focal','center':'smooth_l1','region':'ordered quad smooth_l1, weight 5'}})
    base=load_model(args.base,'cuda').base.eval().requires_grad_(False)
    cache=args.output/'feature_cache.pt'
    records=to_device(make_cache(args.dataset,base,cache,'cuda'),'cuda')
    train=[i for i,r in enumerate(records) if r['row']['split']=='train']
    cal=[i for i,r in enumerate(records) if r['row']['split']=='calibration']
    test=[i for i,r in enumerate(records) if r['row']['split']=='test']
    geom=ceiling(records,train+cal) # No model selection on final test.
    write(args.output/'geometry_ceiling_development.json',geom);print('GEOMETRY CEILING',json.dumps(geom),flush=True)
    model=BandDecoder().cuda()
    with (args.output/'history.jsonl').open('w') as log:
        tiny=train[:8]
        tiny_hist=train_band(model,records,tiny,args.tiny_steps,8,1e-3,log,'tiny_overfit',args.seed)
        tiny_preds=predictions(model,records,tiny)
        tiny_metrics=evaluate(records,tiny,tiny_preds,.25)
        write(args.output/'tiny_overfit.json',tiny_metrics)
        torch.save(model.state_dict(),args.output/'tiny_band.pth')
        print('TINY RESULT',json.dumps(tiny_metrics),flush=True)
        # Reset weights so the independent pilot has a documented fresh start.
        torch.manual_seed(args.seed);model=BandDecoder().cuda()
        train_band(model,records,train,args.steps,args.batch_size,7e-4,log,'band_train',args.seed,
                   eval_every=300,cal=cal,save=args.output/'best_band.pth')
        best=torch.load(args.output/'best_band.pth',map_location='cuda',weights_only=False)
        model.load_state_dict(best['band']);threshold=best['threshold']
        if args.atom_steps:train_atom(base,records,train,cal,args.atom_steps,min(4,args.batch_size),log,args.output)
    # Threshold and checkpoint are locked using calibration data before final test.
    test_preds=predictions(model,records,test)
    before=evaluate(records,test,test_preds,threshold,'baseline_atoms')
    after=evaluate(records,test,test_preds,threshold,'finetuned_atoms' if args.atom_steps else 'baseline_atoms')
    results=dict(status='completed',splits={'train':len(train),'calibration':len(cal),'test':len(test)},
                 best_band_step=best['step'],threshold=threshold,
                 calibration=best['calibration'],tiny=tiny_metrics,
                 geometry_ceiling_test=ceiling(records,test),test_before_atom_finetune=before,test=after,
                 limitations=['Small same-renderer synthetic pilot; no real-image validation.',
                              'Shared backbone/encoder frozen; box and band branches trained independently.',
                              'Band representation is a tapered region quadrilateral, not a fine ink segmentation mask.',
                              'Aromatic chemical bonds evaluated in their drawn Kekule single/double form.'])
    write(args.output/'results.json',results)
    visualize(records,test,test_preds,threshold,args.dataset,args.output)
    serial={records[i]['row']['image']:[dict(label=p['label'],score=p['score'],quad=p['quad'].tolist()) for p in test_preds[i]] for i in test}
    write(args.output/'test_predictions.json',serial)
    print('FINAL RESULTS',json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
