#!/usr/bin/env python3
"""Import five native transfer snapshots for re-review without claiming fresh confirmation."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--storage',type=Path,default=ROOT/'webapp_data')
    args=p.parse_args();storage=args.storage.resolve();examples=storage/'examples'
    imports=storage/'fixed_data/imported_snapshots';examples.mkdir(parents=True,exist_ok=True);imports.mkdir(parents=True,exist_ok=True)
    source=ROOT/'runs/pair_human_only_all_20261026'
    for directory in ['results_data','fixed_data/training_pairs','pasted_examples']:(storage/directory).mkdir(parents=True,exist_ok=True)
    records=[]
    for row in json.loads((source/'manifest.json').read_text()):
        if row['source']!='webapp_human_fix' or row.get('augmentation',{}).get('kind')!='native':continue
        image=source/row['image'];truth=source/row['truth'];ann=json.loads(truth.read_text());w,h=ann['image_size']
        sha=hashlib.sha256(image.read_bytes()).hexdigest();stem=Path(row['file_name']).stem
        if not (examples/image.name).exists():shutil.copy2(image,examples/image.name)
        if not (imports/f'{sha}.source.json').exists():shutil.copy2(truth,imports/f'{sha}.source.json')
        atoms=[];charges=[];bonds=[]
        for atom in ann['atoms']:
            x,y,bw,bh=atom['bbox'];box=[x,y,x+bw,y+bh]
            atoms.append(dict(uid=atom.get('source_uid') or f'atom:{atom["index"]}',source='atom',
                              label=atom.get('source_label') or atom['symbol'],score=1.,bbox=box,
                              center=[x+bw/2,y+bh/2]))
        for i,charge in enumerate(ann.get('charges',[])):
            label=str(charge.get('canonical_label',charge.get('label','')))
            try:label=f'{int(label):+d}'
            except ValueError:pass
            x0,y0,x1,y1=charge['bbox']
            charges.append(dict(uid=charge.get('uid') or f'charge:{i}',source='charge',label=label,
                                score=1.,bbox=[x0,y0,x1,y1],center=[(x0+x1)/2,(y0+y1)/2]))
        labels={0:'other',1:'single',2:'double',3:'triple',4:'aromatic',5:'wedge',6:'dash'}
        for i,bond in enumerate(ann['bonds']):
            line=bond['line'];bonds.append(dict(uid=bond.get('source_uid') or f'bond:{i}',source='bond',
                label=labels[bond['category_id']],score=1.,bbox=line,
                center=[(line[0]+line[2])/2,(line[1]+line[3])/2]))
        seed=dict(schema='i2bl_detection_ground_truth_v1',revision=0,saved_at_utc=None,source_stem=stem,
                  review_status='imported_unreviewed',
                  image=dict(file=image.name,sha256=sha,size=dict(width=w,height=h)),
                  atoms=atoms,charges=charges,bonds=bonds,ocr=ann.get('ocr') or {'items':[]},
                  provenance=dict(source='transfer_training_snapshot',source_annotation=str(truth),
                    source_annotation_sha256=hashlib.sha256(truth.read_bytes()).hexdigest(),
                    original_webapp_image_sha256=row.get('source_hash'),
                    source_image_sha256=sha,latest_webapp_revision_verified=False,
                    original_snapshot=str(imports/f'{sha}.source.json')))
        seedpath=examples/(stem+'.annotation.json')
        if not seedpath.exists():seedpath.write_text(json.dumps(seed,ensure_ascii=False,indent=2))
        records.append(dict(name=image.name,image_sha256=sha,atoms=len(atoms),charges=len(charges),bonds=len(bonds)))
    (storage/'import_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    classpath=storage/'classes.json'
    if not classpath.exists():shutil.copy2(source/'annotations/classes.json',classpath)
    print(json.dumps(dict(storage=str(storage),native_examples=len(records),samples=records),indent=2))

if __name__=='__main__':main()
