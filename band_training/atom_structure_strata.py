"""Check structural error enrichment on ALL GT atoms, including correct images."""
import argparse,gzip,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.spatial import cKDTree


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={}
    for split in ['train','val']:
        stats=defaultdict(Counter);images=defaultdict(Counter);examples=[]
        with gzip.open(a.evaluation/split/'prediction_cache/predictions.jsonl.gz','rt') as f:
            for d in map(json.loads,f):
                atoms=d['ann']['atoms'];m={int(k):v for k,v in d['atom_mapping'].items()};covered=set(m.values())
                incident=defaultdict(list)
                for b in d['ann']['bonds']:
                    incident[b['begin']].append((b['end'],b['category_id']-1));incident[b['end']].append((b['begin'],b['category_id']-1))
                has_terminal=False;terminal_missed=[]
                boxes=np.asarray([x['bbox'] for x in atoms]);centers=boxes[:,:2]+boxes[:,2:]/2
                near=cKDTree(centers).query(centers,k=2)[0][:,1] if len(atoms)>1 else np.full(len(atoms),np.inf)
                for j,t in enumerate(atoms):
                    miss=j not in covered;size=min(t['bbox'][2:]);cl=t['category_id'];edges=incident[j]
                    group='size<8' if size<8 else 'size8-12' if size<12 else 'size12-20' if size<20 else 'size>=20'
                    density='neighbor<10' if near[j]<10 else 'neighbor10-20' if near[j]<20 else 'neighbor20-30' if near[j]<30 else 'neighbor>=30_or_isolated'
                    for name in ['all_atoms',group,density]:stats[name].update(gt=1,fn=int(miss))
                    if cl==1:
                        if any(label==2 for _,label in edges):
                            name='C_terminal_triple' if len(edges)==1 else 'C_internal_triple'
                            if len(edges)==1:
                                has_terminal=True
                                if miss:terminal_missed.append(j)
                        elif len(edges)==1:name='C_terminal_nontriple'
                        elif len(edges)==2:name='C_degree2_nontriple'
                        else:name='C_other_degree'
                        stats[name].update(gt=1,fn=int(miss));stats['all_C'].update(gt=1,fn=int(miss))
                bucket='has_terminal_triple_C' if has_terminal else 'without_terminal_triple_C'
                fp=sum(i not in m for i in range(len(d['atoms'])));fn=len(atoms)-len(covered)
                images[bucket].update(images=1,atom_error_images=int(bool(fp or fn)),fp=fp,fn=fn,terminal_C_fn=len(terminal_missed))
                if terminal_missed:examples.append(dict(sample_id=d['sample_id'],missed_terminal_triple_C=terminal_missed))
        result[split]=dict(atom_strata={k:dict(c,fn_rate=c['fn']/c['gt']) for k,c in stats.items()},
            image_strata={k:dict(c,error_rate=c['atom_error_images']/c['images']) for k,c in images.items()},terminal_examples=examples)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps({s:{'atoms':{k:v for k,v in r['atom_strata'].items() if k.startswith('C_')},'images':r['image_strata']} for s,r in result.items()},indent=2))


if __name__=='__main__':main()
