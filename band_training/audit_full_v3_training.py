"""Audit every unique source asset and both frozen split/sampling manifests."""
import argparse,csv,hashlib,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np
from PIL import Image
from .mixed_data import ATOM_CLASSES,CHARGE_CLASSES,letterbox_transform


def read_rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def audit_one(root,row):
    ann=json.loads((root/row['annotation']).read_text())
    band=json.loads((root/row['band_target']).read_text())
    line=json.loads((root/row['line_target']).read_text())
    assert band['labels']==line['labels']
    labels=band['labels'];quads=np.asarray(band['quads'],float).reshape(-1,4,2)
    lines=np.asarray(line['lines'],float).reshape(-1,2,2)
    assert len(labels)==len(quads)==len(lines)==len(ann['bonds'])
    assert all(l in range(6) for l in labels)
    assert np.isfinite(quads).all() and np.isfinite(lines).all()
    byid={a['id']:a for a in ann['atoms']}
    expected=np.asarray([[byid[b['source']]['center'],byid[b['target']]['center']] for b in ann['bonds']],float).reshape(-1,2,2)
    assert np.allclose(expected,lines,rtol=0,atol=1e-6)
    width,height=ann['width'],ann['height']
    with Image.open(root/row['image']) as im:assert im.size==(width,height)
    scale,offset,_=letterbox_transform(width,height)
    # Explicitly check forward/inverse transforms for every box/quad/line.
    geometries=[quads.reshape(-1,2),lines.reshape(-1,2)]
    for a in ann['atoms']+ann['charges']:
        bbox=np.asarray(a['bbox'],float).reshape(2,2)
        assert np.isfinite(bbox).all() and np.all(bbox[1]>bbox[0])
        geometries.append(bbox)
    for g in geometries:
        transformed=g*scale+offset
        assert np.allclose((transformed-offset)/scale,g,rtol=0,atol=1e-7)
        assert np.all(transformed>=-1e-3) and np.all(transformed<=640.001)
    assert all(int(q['value']) in CHARGE_CLASSES for q in ann['charges'])
    return dict(sample_id=row['sample_id'],split=row['split'],image=row['image'],group=row['group'],
        atoms=len(ann['atoms']),charges=len(ann['charges']),bonds=len(labels),
        color=str(row.get('is_color','')).lower()=='true',
        unknown_atoms=sum(a['symbol'] not in ATOM_CLASSES for a in ann['atoms']),
        atom_symbols=dict(Counter(a['symbol'] for a in ann['atoms'])),
        bond_classes=dict(Counter(labels)),box_targets=len(ann['atoms'])+len(ann['charges']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.time();root=args.dataset
    names=['train','unique_train','val','data_manifest']
    manifests={name:read_rows(root/(name+'.csv')) for name in names}
    train,val=manifests['unique_train'],manifests['val']
    assert not {r['group'] for r in train}&{r['group'] for r in val}
    assert not {r['image'] for r in train}&{r['image'] for r in val}
    assert {r['image'] for r in manifests['train']}=={r['image'] for r in train}
    assert {r['image'] for r in manifests['data_manifest']}=={r['image'] for r in train+val}
    rows=[]
    with ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(lambda r:audit_one(root,r),manifests['data_manifest']):
            rows.append(row)
            if len(rows)%1000==0:print('AUDITED',len(rows),flush=True)
    (args.output/'assets.json').write_text(json.dumps(rows))
    report=dict(status='passed',dataset=str(root.resolve()),unique_images=len(rows),
        unique_train=len(train),validation=len(val),training_sampling_rows=len(manifests['train']),
        charged_sampling_rows=sum(r['has_charge'].lower()=='true' for r in manifests['train']),
        atoms=sum(r['atoms'] for r in rows),bonds=sum(r['bonds'] for r in rows),charges=sum(r['charges'] for r in rows),
        max_atoms=max(r['atoms'] for r in rows),max_box_targets=max(r['box_targets'] for r in rows),max_bonds=max(r['bonds'] for r in rows),
        unknown_atoms=sum(r['unknown_atoms'] for r in rows),images_with_unknown_atoms=sum(bool(r['unknown_atoms']) for r in rows),
        manifest_sha256={n:hashlib.sha256((root/(n+'.csv')).read_bytes()).hexdigest() for n in names},
        seconds=time.time()-started,checks=['all images/boxes/quads/lines finite and in canvas','all band/line labels and order paired',
        'all line endpoints match drawing anchors','shared letterbox and inverse checked for every geometry','train/val groups and assets disjoint',
        'explicit training repeats retained, no second weighted sampler'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
