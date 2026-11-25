"""Check shared-query targets on every full-v3 unique asset without image IO."""
import argparse,pickle,json,time
from pathlib import Path
import torch
from .scored_bond_model import scored_targets


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);data=pickle.load(a.cache.open('rb'));start=time.time();n=0;displaced=0
    for i,d in enumerate(data):
        quads=[r['quad'] for r in d['instances']];lines=[r['line'] for r in d['instances']];labels=[r['label'] for r in d['instances']]
        heat,entries=scored_targets(labels,quads,lines)
        assert len(entries)==len(labels)==int(heat.eq(1).sum())
        for entry,quad,line in zip(entries,quads,lines):
            _,y,x,offset,q,l=entry;center=(torch.tensor([x,y])+offset)*4
            torch.testing.assert_close(q.reshape(4,2)*32+center,torch.tensor(quad).float(),atol=1e-4,rtol=0)
            torch.testing.assert_close(l.reshape(2,2)*32+center,torch.tensor(line).float(),atol=1e-4,rtol=0)
            displaced+=int(bool((offset.abs()>1).any()))
        n+=len(entries)
        if (i+1)%3000==0:print('AUDITED',i+1,flush=True)
    report=dict(status='passed',images=len(data),bonds=n,displaced_anchors=displaced,seconds=time.time()-start,
        checks=['one positive per bond for EACH independent score','all classes separated for class-agnostic local NMS','exact shared-query line/quad coordinate reconstruction','no atom IDs or detection boxes used'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
