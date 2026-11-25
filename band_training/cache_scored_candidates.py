"""One GPU pass: cache pre-local-peak candidates and dense GT-only diagnostics."""
import argparse,gzip,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train import load_model,decode_atoms
from .continue_indigo import loader,match_boxes,write
from .full_v3_data import FullV3Dataset
from .scored_bond_model import ScoredBondDecoder,decode_scored_bonds
from .train_scored_bonds import ScoredJointModel
from .geometry import LineEndpointIndex
from .cache_final_predictions import encode


def dense_diagnostic(head,bi,meta):
    score=head['score_logits'][bi,0].sigmoid();h,w=score.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=score.device),torch.arange(w,device=score.device),indexing='ij')
    center=(torch.stack((xx,yy),0)+head['offset'][bi])*4
    lines=(head['endpoints'][bi].reshape(2,2,h,w)*32+center[None]).permute(2,3,0,1).reshape(-1,2,2).cpu().numpy()
    scores=score.flatten().cpu().numpy();peaks=(score==F.max_pool2d(score[None,None],3,1,1)[0,0]).flatten().cpu().numpy()
    pairs=LineEndpointIndex([a['bbox'] for a in meta['ann']['atoms']]).link_many(lines)
    best={};best_peak={}
    for q,(pair,s) in enumerate(zip(pairs,scores)):
        if pair is None:continue
        key=tuple(sorted(pair))
        if s>best.get(key,(-1,None))[0]:best[key]=(float(s),q)
        if peaks[q] and s>best_peak.get(key,(-1,None))[0]:best_peak[key]=(float(s),q)
    records=[]
    for b in meta['ann']['bonds']:
        key=tuple(sorted((b['begin'],b['end'])))
        record=dict(pair=key,label=b['category_id']-1)
        for name,table in [('dense',best),('peak',best_peak)]:
            s,q=table.get(key,(None,None));record[name+'_score']=s
            if q is not None:record[name+'_query']=q;record[name+'_line']=lines[q].tolist()
        records.append(record)
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset-split',default='val');p.add_argument('--workers',type=int,default=4);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=s['config'];device=torch.device('cuda:0')
    base=load_model(Path(c['base']),device).base;base.load_state_dict(s['base']);decoder=ScoredBondDecoder().to(device);decoder.load_state_dict(s['decoder'])
    model=ScoredJointModel(base,decoder).to(device).eval();data=FullV3Dataset(c['dataset'],a.dataset_split,endpoint_order=c.get('endpoint_order','canonical'))
    previous=Path('runs/scored_bond_v3_comparison_20261118/missing_diagnosis/samples.json')
    diagnostic_ids={r['sample_id'] for r in json.loads(previous.read_text())};n=0;started=time.time()
    with torch.no_grad(),gzip.open(a.output/'candidates.jsonl.gz.tmp','wt',compresslevel=3) as out:
        for images,metas in loader(data,c['batch_size'],a.workers):
            box,heads=model(images.to(device))
            for bi,meta in enumerate(metas):
                ap=decode_atoms({k:v[bi:bi+1] for k,v in box.items() if torch.is_tensor(v) and v.ndim})
                atoms=[v for v in ap if v['label']<13];charges=[v for v in ap if v['label']>=13]
                decoded=decode_scored_bonds(heads,bi,.05,.05,apply_line_peaks=False)
                row=dict(sample_id=meta['record']['sample_id'],record=meta['record'],ann=meta['ann'],gt_lines=meta['lines'],gt_quads=meta['quads'],
                         atoms=atoms,charges=charges,atom_mapping=match_boxes(atoms,meta['ann']['atoms']),
                         charge_detection_mapping=match_boxes(charges,meta['ann']['charges'],min_iou=.25),independent_heads=decoded)
                if row['sample_id'] in diagnostic_ids:row['dense_diagnostic']=dense_diagnostic(heads['line'],bi,meta)
                out.write(json.dumps(row,default=encode)+'\n');n+=1
            if n%128<len(metas):print('CACHED',n,'/',len(data),flush=True)
    (a.output/'candidates.jsonl.gz.tmp').replace(a.output/'candidates.jsonl.gz')
    report=dict(status='completed',images=n,checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                dataset=c['dataset'],split=a.dataset_split,minimum_scores=.05,includes_nonpeak_line_candidates=True,seconds=time.time()-started)
    write(a.output/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
