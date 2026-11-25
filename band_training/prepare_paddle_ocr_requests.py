"""Export frozen crops without GT text for isolated PaddleOCR inference."""
import argparse,gzip,hashlib,json
from pathlib import Path


def request_key(image,box):
    payload=json.dumps([str(Path(image).resolve()),[float(v) for v in box]],separators=(',',':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--easyocr',type=Path)
    p.add_argument('--include-gt-diagnostics',action='store_true')
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with gzip.open(a.cache,'rt') as f:data=[json.loads(line) for line in f]
    images={d['sample_id']:d['record']['image'] for d in data}
    requests={}
    if a.easyocr:
        crops=json.loads((a.easyocr/'ocr_crops.json').read_text())
    else:
        import numpy as np
        from .mixed_data import letterbox_transform
        crops=[]
        for d in data:
            raw=json.loads((a.dataset/d['record']['annotation']).read_text())
            scale,offset,_=letterbox_transform(raw['width'],raw['height'])
            for atom in d['atoms']:
                if atom['label']!=0:continue
                x,y,w,h=atom['bbox'];box=np.r_[(np.array([x,y])-offset)/scale,(np.array([x+w,y+h])-offset)/scale]
                crops.append(dict(sample_id=d['sample_id'],box=box.tolist()))
            if a.include_gt_diagnostics:
                for prepared,original in zip(d['ann']['atoms'],raw['atoms']):
                    if prepared['category_id']==0:crops.append(dict(sample_id=d['sample_id'],box=original['bbox']))
    for crop in crops:
        image=str((a.dataset/images[crop['sample_id']]).resolve());box=crop['box'];key=request_key(image,box)
        requests[key]=dict(key=key,image=image,box=[float(v) for v in box])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(list(requests.values()),indent=2))
    print('REQUESTS',len(requests),'GT labels exported: false')


if __name__=='__main__':main()
