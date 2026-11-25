"""Verify image-coordinate target order preserves the full GT graph ceiling."""
import argparse,json,pickle,time
from pathlib import Path
from collections import Counter
import numpy as np
from .full_v3_data import orient_targets
from .geometry import assemble_band_lines,edge_key


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=pickle.load(a.cache.open('rb'));stats=Counter();failures=[];started=time.time()
    for i,d in enumerate(rows):
        labels=[b['label'] for b in d['instances']];quads=[b['quad'] for b in d['instances']];lines=[b['line'] for b in d['instances']]
        q,l=orient_targets(labels,quads,lines)
        stats['raw_band_line_reversed']+=sum(np.dot(np.mean(b[2:],0)-np.mean(b[:2],0),np.asarray(e)[1]-np.asarray(e)[0])<0 for b,e in zip(quads,lines))
        for label,old,new in zip(labels,lines,l):
            if label in [4,5]:assert np.array_equal(old,new)
            assert {tuple(p) for p in old}=={tuple(p) for p in new}
        original=assemble_band_lines(d['boxes'],d['instances'],line_association='exclusive_box')
        truth=Counter(edge_key(b['begin'],b['end'],b['label']) for b in original['bonds'])
        instances=[dict(label=label,quad=quad,line=line) for label,quad,line in zip(labels,q,l)]
        for method,radius in [('exclusive_box',0.),('exclusive_box',2.),('exclusive_box_noisy',2.)]:
            graph=assemble_band_lines(d['boxes'],instances,line_association=method,endpoint_cluster_radius=radius)
            edges=Counter(edge_key(b['begin'],b['end'],b['label']) for b in graph['bonds'])
            ok=edges==truth and not graph['unlinked_band_indices'] and len(graph['bonds'])==len(instances)
            stats[method+'_exact_radius_'+str(radius)]+=int(ok)
            if not ok:failures.append(dict(sample_id=d['sample_id'],method=method,radius=radius))
        stats['images']+=1;stats['bonds']+=len(instances)
        if (i+1)%3000==0:print('CHECKED',i+1,flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='passed' if not failures else 'failed',stats={k:int(v) for k,v in stats.items()},failures=failures,seconds=time.time()-started,
        note='Compared canonical targets against validated raw geometry; ordinary edges unordered, wedge/dash directed. Charge ownership unchanged.')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
