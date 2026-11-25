"""Trace the 148 region-unmatched missing typed edges through threshold and NMS."""
import argparse,json,gzip,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Subset
from .train import load_model
from .continue_indigo import JointModel,loader,write
from .band_line_model import BandLineDecoder
from .full_v3_data import FullV3Dataset
from .model import decode
from .geometry import polygon_iou,suppress


def compatible(instances,quad,line):
    q=np.asarray(quad);l=np.asarray(line);length=max(4.,float(np.linalg.norm(l[1]-l[0])));result=[]
    for i,p in enumerate(instances):
        pq=np.asarray(p['quad']);pl=np.asarray(p['line']);cd=float(np.linalg.norm(pq.mean(0)-q.mean(0)))
        if cd>length*.8+8:continue
        iou=polygon_iou(pq,q);ep=float(min(np.linalg.norm(pl-l,axis=1).mean(),np.linalg.norm(pl-l[::-1],axis=1).mean()))
        if iou>=.1 or (cd<max(4,.2*length) and ep<max(8,.6*length)):
            result.append(dict(index=i,label=p['label'],score=p['score'],iou=iou,center_distance=cd,endpoint_error=ep))
    return sorted(result,key=lambda x:-(x['iou']+.001*x['score']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    wanted={r['sample_id']:[x['gt_bond'] for x in r['missing'] if x['reason']=='no_geometrically_matched_region_at_threshold'] for r in json.loads((a.diagnosis/'samples.json').read_text())}
    wanted={k:v for k,v in wanted.items() if v}
    with gzip.open(a.cache,'rt') as f:cached={d['sample_id']:d for d in map(json.loads,f) if d['sample_id'] in wanted}
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=s['config'];device=torch.device('cuda:0')
    base=load_model(Path(c['base']),device).base;base.load_state_dict(s['base']);band=BandLineDecoder().to(device);band.load_state_dict(s['band'])
    model=JointModel(base,band).to(device).eval();val=FullV3Dataset(c['dataset'],'val',endpoint_order=c.get('endpoint_order','canonical'))
    ids=[i for i,r in enumerate(val.records) if r['sample_id'] in wanted];counts=Counter();rows=[]
    with torch.no_grad():
        for images,metas in loader(Subset(val,ids),8,4):
            _,out=model(images.to(device))
            for bi,meta in enumerate(metas):
                sid=meta['record']['sample_id'];raw=decode(out,bi,minimum=.01)
                high=[p for p in raw if p['score']>=.2];kept=cached[sid]['instances'];heat=out['heat'][bi].sigmoid().float().cpu().numpy()
                for j in wanted[sid]:
                    q=meta['quads'][j];line=meta['lines'][j];label=meta['labels'][j]
                    A=compatible(kept,q,line);B=compatible(high,q,line);C=compatible(raw,q,line)
                    _,y,x,_,_=meta['band'][1][j];patch=heat[:,max(0,y-2):y+3,max(0,x-2):x+3]
                    best=float(patch.max());same=float(patch[label].max())
                    if A:reason='retained_region_exists_but_one_to_one_diagnostic_pairing_ambiguous'
                    elif B:reason='candidate_removed_by_quad_nms'
                    elif C:reason='geometrically_compatible_candidate_below_0.2_score'
                    elif best>=.2:reason='local_heat_response_but_no_geometrically_compatible_candidate'
                    else:reason='weak_local_detection_heat_below_0.2'
                    counts[reason]+=1
                    rows.append(dict(sample_id=sid,gt_bond=j,gt_label=label,reason=reason,expected_class_local_peak=same,any_class_local_peak=best,
                        after_nms=A[:2],before_nms_above_0p2=B[:2],raw_above_0p01=C[:2],
                        raw_candidates=len(raw),gt_bond_count=len(meta['quads'])))
                print('PROBED',sid,len(wanted[sid]),flush=True)
    assert len(rows)==148
    report=dict(images=len(ids),missing_edges=len(rows),counts=dict(counts),scope='Diagnostic subset only. Region compatibility is label-free IoU>=0.1 or tight center+endpoint geometry. A heat response alone does not prove a correct bond detection; broad compatibility can be ambiguous.',
        no_training_or_production_changes=True)
    write(a.output/'report.json',report);write(a.output/'samples.json',rows);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
