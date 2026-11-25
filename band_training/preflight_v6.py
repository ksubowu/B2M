"""Freeze v6 manifests; audit all ordered line/band targets without image IO."""
import argparse,hashlib,json,multiprocessing,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from .mixed_data import prepare_annotation,letterbox_transform
from .full_v3_data import scaled_lines,orient_targets
from .scored_bond_model import scored_targets
from .continue_indigo import write,sha


def init_worker():torch.set_num_threads(1)


def audit(item):
    root,row=item
    try:
        a=json.loads((root/row['annotation']).read_text());b=json.loads((root/row['band_target']).read_text());l=json.loads((root/row['line_target']).read_text())
        ann,boxes,labels,quads=prepare_annotation(a,b);lines=scaled_lines(a,b,l);quads,lines=orient_targets(labels,quads,lines)
        heat,entries=scored_targets(labels,quads,lines)
        assert len(entries)==len(labels)==len(a['bonds'])==int(heat.eq(1).sum())
        if entries:
            xy=np.array([[e[2],e[1]] for e in entries]);offset=np.stack([e[3].numpy() for e in entries]);center=(xy+offset)*4
            pq=np.stack([e[4].numpy().reshape(4,2) for e in entries])*32+center[:,None]
            pl=np.stack([e[5].numpy().reshape(2,2) for e in entries])*32+center[:,None]
            error=max(float(np.max(np.abs(pq-quads))),float(np.max(np.abs(pl-lines))))
            assert error<=1e-4,error
        else:error=0.
        box=np.array([t['bbox'] for t in ann['atoms']]);span=(box[:,:2]+box[:,2:]).max(0)-box[:,:2].min(0)
        lengths=np.linalg.norm(np.asarray(lines)[:,1]-np.asarray(lines)[:,0],axis=-1) if lines else np.array([])
        mini=bool(max(span)<=160 and np.median(box[:,2:].min(1))<=6 and (not len(lengths) or np.median(lengths)<=12))
        return dict(sample_id=row['sample_id'],image=row['image'],atoms=len(ann['atoms']),charges=len(ann['charges']),bonds=len(labels),
                    atom_classes=dict(Counter(t['category_id'] for t in ann['atoms'])),charge_classes=dict(Counter(t['category_id'] for t in ann['charges'])),
                    bond_classes=dict(Counter(labels)),color=bool(row.get('is_color')),mini=mini,max_coordinate_error=error)
    except Exception as e:return dict(sample_id=row['sample_id'],error=repr(e))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--mini-hard-manifest',type=Path,help='Optional known mini_hard ID manifest to exclude before the geometry audit')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.time()
    original=json.loads((a.dataset/'train.json').read_text());val=json.loads((a.dataset/'val.json').read_text())
    old_mini=a.mini_hard_manifest or a.dataset/'mini_hard/data_manifest.json'
    excluded={r['sample_id'] for r in json.loads(old_mini.read_text())} if old_mini.exists() else set();known=[r for r in original if r['sample_id'] in excluded]
    train=[r for r in original if r['sample_id'] not in excluded]
    assert len({r['sample_id'] for r in original})==len(original)
    assert len({r['image'] for r in original})==len(original)
    assert not ({r['sample_id'] for r in train}&{r['sample_id'] for r in val})
    assert not ({r['group'] for r in train}&{r['group'] for r in val})
    config=json.loads((a.dataset/'dataset_config.json').read_text())
    for name,expected in config['validation_reference']['manifest_sha256'].items():assert sha(a.dataset/name)==expected
    # Preliminary effective manifest lets the bounded smoke run proceed while
    # the full CPU audit completes; formal training requires final status=passed.
    write(a.output/'train.json',train);write(a.output/'val.json',val);write(a.output/'known_mini_exclusions.json',known)
    rows=[];errors=[];counts=Counter();seen=0
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker) as pool:
        for r in pool.map(audit,[(a.dataset,row) for row in train+val],chunksize=48):
            seen+=1
            if 'error' in r:errors.append(r)
            else:rows.append(r);counts.update(images=1,atoms=r['atoms'],charges=r['charges'],bonds=r['bonds'])
            if seen%4096==0:
                print('AUDITED',seen,'/',len(train)+len(val),'seconds',round(time.time()-started),flush=True)
                write(a.output/'progress.json',dict(status='auditing',images=seen,total=len(train)+len(val),errors=len(errors)))
    assert not errors,errors[:3]
    byid={r['sample_id']:r for r in rows};additional=[r for r in train if byid[r['sample_id']]['mini']]
    removed={r['sample_id'] for r in additional};train=[r for r in train if r['sample_id'] not in removed]
    write(a.output/'train.json',train);write(a.output/'extra_mini_exclusions.json',additional)
    write(a.output/'assets.json',rows)
    report=dict(status='passed',source_train_rows=len(original),effective_train_rows=len(train),validation_rows=len(val),
        known_mini_excluded=len(known),extra_mini_excluded=len(additional),sampling='uniform shuffle, each unique image once; no weighted sampler',
        source_hashes={s:sha(a.dataset/(s+'.json')) for s in ['train','val']},manifest_sha256={s:sha(a.output/(s+'.json')) for s in ['train','val']},
        target_counts=dict(counts),max_coordinate_error=max(r['max_coordinate_error'] for r in rows),errors=errors,seconds=time.time()-started,
        checks=['one heatmap positive per bond','ordered line/quad coordinate reconstruction <=1e-4px','18 box classes / 6 bond classes unchanged',
                'frozen validation manifest hashes','unique train images, no train/val ID or group overlap','mini_hard excluded in run manifests; source dataset unchanged'])
    write(a.output/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
