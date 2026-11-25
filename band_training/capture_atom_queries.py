"""Reproduce original batches and retain pre-threshold/pre-NMS atom queries."""
import argparse,gzip,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.ops import box_convert,nms
from .train import load_model,decode_atoms
from .continue_indigo import loader,write
from .full_v3_data import FullV3Dataset
from .scored_bond_model import ScoredBondDecoder
from .train_scored_bonds import ScoredJointModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--diagnosis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    previous=json.loads((a.evaluation/'val/prediction_cache/report.json').read_text());checkpoint=Path(previous['checkpoint'])
    s=torch.load(checkpoint,map_location='cpu',weights_only=False);c=s['config'];device=torch.device('cuda:0')
    base=load_model(Path(c['base']),device).base;base.load_state_dict(s['base'])
    decoder=ScoredBondDecoder().to(device);decoder.load_state_dict(s['decoder']);model=ScoredJointModel(base,decoder).eval().to(device)
    reports={};started=time.time();count=0
    with torch.no_grad(),gzip.open(a.output/'queries.jsonl.gz.tmp','wt',compresslevel=3) as target:
        for split in ['train','val']:
            failures=json.loads((a.diagnosis/(split+'_samples.json')).read_text());wanted={r['sample_id'] for r in failures}
            with gzip.open(a.evaluation/split/'prediction_cache/predictions.jsonl.gz','rt') as f:
                frozen={d['sample_id']:d for d in map(json.loads,f) if d['sample_id'] in wanted}
            ds=FullV3Dataset(c['dataset'],split);ds.records=json.loads((a.evaluation/split/'prediction_cache/evaluated_manifest.json').read_text())
            bs=c['batch_size'];blocks=sorted({i//bs for i,r in enumerate(ds.records) if r['sample_id'] in wanted})
            indices=[i for block in blocks for i in range(block*bs,min((block+1)*bs,len(ds)))]
            print('BATCH_CONTEXT',split,'original images',len(indices),'failed images',len(wanted),flush=True)
            checked=0;max_error=0.
            for images,metas in loader(Subset(ds,indices),bs,4):
                # Identical encoder and box-decoder path, original batch members
                # and batch shape; the independent bond decoder is not needed.
                features=model.features(images.to(device));out=base.box_decoder(features)
                for bi,meta in enumerate(metas):
                    sid=meta['record']['sample_id']
                    if sid not in wanted:continue
                    before=frozen[sid];logits=out['pred_logits'][bi];prob=logits.float().sigmoid();score,label=prob.max(-1)
                    xyxy=box_convert(out['pred_boxes'][bi],'cxcywh','xyxy')*640
                    active=torch.where((score>.25)&(label<13))[0];kept=active[nms(xyxy[active],score[active],.4)].tolist()
                    now=decode_atoms(dict(pred_logits=out['pred_logits'][bi:bi+1],pred_boxes=out['pred_boxes'][bi:bi+1]))
                    now=[x for x in now if x['label']<13];cached=before['atoms'];assert len(now)==len(cached),(split,sid,'count drift')
                    for x,y in zip(now,cached):
                        assert x['label']==y['label'],(sid,'label drift')
                        delta=float(np.max(np.abs(np.array(x['bbox'])-y['bbox'])));max_error=max(max_error,delta)
                        assert delta<=.02 and abs(x['score']-y['score'])<=1e-4,(sid,'prediction drift',delta)
                    raw=xyxy.detach().float().cpu().numpy();clipped=raw.clip(0,640);boxes=np.c_[clipped[:,:2],clipped[:,2:]-clipped[:,:2]]
                    d=dict(split=split,sample_id=sid,boxes=boxes.tolist(),xyxy_unclipped=raw.tolist(),class_scores=prob.cpu().tolist(),
                        top_class=label.cpu().tolist(),top_score=score.cpu().tolist(),kept_atom_queries=kept,
                        cached_atoms=cached,atom_mapping=before['atom_mapping'],ann_atoms=before['ann']['atoms'])
                    target.write(json.dumps(d)+'\n');checked+=1;count+=1
                if count%64<len(metas):print('CAPTURED',split,count,'seconds',round(time.time()-started),flush=True)
            assert checked==len(wanted);reports[split]=dict(failed_images=checked,context_images=len(indices),max_box_difference=max_error)
    (a.output/'queries.jsonl.gz.tmp').replace(a.output/'queries.jsonl.gz')
    result=dict(status='completed',checkpoint=str(checkpoint),checkpoint_sha256=previous['checkpoint_sha256'],splits=reports,seconds=time.time()-started,
        scope='Raw probabilities, boxes and NMS provenance. Original batch context preserved; all failed-image atom predictions checked against frozen evaluation.')
    write(a.output/'report.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
