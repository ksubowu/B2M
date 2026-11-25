"""Posthoc charge diagnostics; does not modify training weights or thresholds."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import box_convert,nms
from PIL import Image,ImageDraw
from .continue_indigo import JointModel,load_model,loader,match_boxes,iou_matrix,write
from .mixed_data import MixedIndigoDataset
from .model import BandDecoder
from .train import decode_atoms,ROOT

def main():
    torch.set_num_threads(2)
    run=ROOT/'runs/indigo_mixed_continue_20261113';out=run/'charge_diagnostic';out.mkdir(exist_ok=True)
    config=json.loads((run/'config.json').read_text());device=torch.device('cuda:0')
    base=load_model(Path(config['base']),device).base
    state=torch.load(run/'best_joint.pt',map_location='cpu',weights_only=False)
    base.load_state_dict(state['base']);band=BandDecoder().to(device);band.load_state_dict(state['band'])
    model=JointModel(base,band).to(device).eval()
    ds=MixedIndigoDataset(config['dataset'],'val');thresholds=[.25,.1,.05,.01]
    counts={str(t):Counter() for t in thresholds};independent={str(t):Counter() for t in thresholds}
    diagnostics=[];gt_sizes=[];gt_classes=Counter();pred_classes=Counter()
    with torch.no_grad():
        for images,metas in loader(ds,8,4):
            images=images.to(device);boxes,_=model(images)
            for bi,m in enumerate(metas):
                gt=m['ann']['charges'];logits=boxes['pred_logits'][bi].sigmoid()
                xyxy=box_convert(boxes['pred_boxes'][bi],'cxcywh','xyxy').clamp(0,1)*640
                xywh=box_convert(xyxy,'xyxy','xywh').cpu().numpy()
                for t in thresholds:
                    pp=decode_atoms({k:v[bi:bi+1] for k,v in boxes.items() if isinstance(v,torch.Tensor)},threshold=t)
                    pp=[p for p in pp if p['label']>=13]
                    counts[str(t)].update(pred=len(pp),tp=len(match_boxes(pp,gt,min_iou=.25)),gt=len(gt))
                    sc,lb=logits[:,13:].max(-1);keep=torch.where(sc>t)[0];keep=keep[nms(xyxy[keep],sc[keep],.4)]
                    iq=[dict(bbox=xywh[i].tolist(),label=int(lb[i])+13,score=float(sc[i])) for i in keep]
                    independent[str(t)].update(pred=len(iq),tp=len(match_boxes(iq,gt,min_iou=.25)),gt=len(gt))
                    if t==.25:pred_classes.update(p['label'] for p in pp)
                if not gt:continue
                gt_classes.update(q['category_id'] for q in gt)
                overlaps=iou_matrix(xywh,[q['bbox'] for q in gt]);matches=[]
                for j,q in enumerate(gt):
                    w,h=q['bbox'][2:];gt_sizes.append([w,h]);i=int(np.argmax(overlaps[:,j]));label=q['category_id']
                    candidates=np.flatnonzero(overlaps[:,j]>=.25)
                    scores=logits[:,label].cpu().numpy()
                    matches.append(dict(gt=q,best_geometry_iou=float(overlaps[i,j]),best_geometry_label=int(logits[i].argmax()),
                        best_geometry_true_charge_score=float(scores[i]),
                        max_true_charge_score_at_iou25=float(scores[candidates].max()) if len(candidates) else 0,
                        max_true_charge_score=float(scores.max())))
                pp=decode_atoms({k:v[bi:bi+1] for k,v in boxes.items() if isinstance(v,torch.Tensor)},threshold=.05)
                pp=[p for p in pp if p['label']>=13]
                diagnostics.append(dict(sample_id=m['record']['sample_id'],matches=matches,predicted_charges_at_005=pp))
                if len(diagnostics)<=6:
                    im=Image.fromarray((images[bi].cpu().permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8'))
                    canvas=Image.new('RGB',(1920,680),'white')
                    for col,title in enumerate(['Input','GT charge boxes','Pred charge boxes threshold .05']):
                        panel=im.copy();d=ImageDraw.Draw(panel)
                        for q in (gt if col==1 else pp if col==2 else []):
                            x,y,w,h=q['bbox'];d.rectangle([x,y,x+w,y+h],outline='red',width=2)
                            d.text((x,max(0,y-12)),str(q.get('category_id',q.get('label'))),fill='red')
                        canvas.paste(panel,(col*640,40));ImageDraw.Draw(canvas).text((col*640+5,8),title,fill='black')
                    canvas.save(out/f'charged_{len(diagnostics):02d}.png')
    def summarize(rows):
        return {t:dict(c,precision=c['tp']/max(1,c['pred']),recall=c['tp']/max(1,c['gt']),f1=2*c['tp']/max(1,c['pred']+c['gt'])) for t,c in rows.items()}
    sizes=np.asarray(gt_sizes)
    report=dict(argmax_decode=summarize(counts),independent_charge_decode=summarize(independent),gt_classes=dict(gt_classes),pred_classes=dict(pred_classes),
        size_percentiles={str(p):np.percentile(sizes,p,axis=0).tolist() for p in [0,25,50,75,100]},charged_samples=diagnostics)
    write(out/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='charged_samples'},indent=2))

if __name__=='__main__':main()
