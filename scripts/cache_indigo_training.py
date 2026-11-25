"""Copy manifest-referenced assets to local NVMe without changing their bytes."""
import argparse,csv,hashlib,json,shutil,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();source=args.source.resolve();style=source.parent;output=args.output.resolve();started=time.time()
dest=output/source.name;dest.mkdir(parents=True,exist_ok=True)
rows={};paths=set();hashes={}
for split in ['train','val']:
    manifest=source/f'{split}.csv';payload=manifest.read_bytes();hashes[split]=hashlib.sha256(payload).hexdigest()
    (dest/manifest.name).write_bytes(payload)
    with manifest.open(newline='') as f:rows[split]=list(csv.DictReader(f))
    for r in rows[split]:
        for key in ['image','annotation','band_target']:paths.add((source/r[key]).resolve())
assert not ({r['group'] for r in rows['train']}&{r['group'] for r in rows['val']}),'Cross-split source group'
for name in ['dataset_config.json','README.md']:
    if (source/name).is_file():shutil.copyfile(source/name,dest/name)
def copy(path):
    target=output/path.relative_to(style);target.parent.mkdir(parents=True,exist_ok=True)
    # Read each source once. Compute a digest on the exact bytes written.
    payload=path.read_bytes();target.write_bytes(payload)
    return str(path.relative_to(style)),len(payload),hashlib.sha256(payload).hexdigest()
records=[]
with ThreadPoolExecutor(max_workers=16) as pool:
    pending=[pool.submit(copy,path) for path in sorted(paths)]
    for future in as_completed(pending):
        records.append(future.result())
        if len(records)%2000==0:print('CACHED',len(records),'/',len(paths),'seconds',round(time.time()-started,1),flush=True)
report=dict(status='passed',source=str(source),dataset=str(dest),manifest_sha256=hashes,
            files=len(records),bytes=sum(r[1] for r in records),seconds=time.time()-started,
            train_rows=len(rows['train']),val_rows=len(rows['val']),
            charge_fraction=sum(r['has_charge'].lower()=='true' for r in rows['train'])/len(rows['train']))
(output/'asset_hashes.json').write_text(json.dumps(records))
(output/'cache_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
