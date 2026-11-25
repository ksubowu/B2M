"""Run with isolated Python 3.11 PaddleOCR 2.x environment; no torch or GT labels."""
import argparse,hashlib,importlib.metadata,json,math,time,unicodedata
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps


def normalized(text):return ''.join(unicodedata.normalize('NFKC',str(text)).split())


def main():
    p=argparse.ArgumentParser();p.add_argument('--requests',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--languages',nargs='+',choices=['en','latin'],default=['en','latin'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cv2
    cv2.setNumThreads(1)
    import paddleocr
    from paddleocr.paddleocr import parse_args
    from tools.infer.predict_rec import TextRecognizer
    engines={}
    for language in a.languages:
        options=parse_args(mMain=False)
        options.use_gpu=False;options.cpu_threads=2;options.enable_mkldnn=False;options.show_log=False
        options.rec_model_dir=str((a.models/(language+'_source')).resolve())
        options.rec_image_shape='3,48,320';options.rec_algorithm='SVTR_LCNet';options.rec_batch_num=8
        dictionary='ppocr/utils/en_dict.txt' if language=='en' else 'ppocr/utils/dict/latin_dict.txt'
        options.rec_char_dict_path=str(Path(paddleocr.__file__).parent/dictionary)
        if not (Path(options.rec_model_dir)/'inference.pdiparams').exists():raise RuntimeError('Missing local Paddle recognition model')
        engines[language]=TextRecognizer(options)
    requests=json.loads(a.requests.read_text());results=[];started=time.time()
    for i,request in enumerate(requests):
        # The request schema deliberately excludes expected strings and graph IDs.
        assert set(request)=={'key','image','box'}
        with Image.open(request['image']) as im:image=im.convert('RGB')
        x1,y1,x2,y2=request['box'];crops=[];specs=[]
        for margin in [1,3]:
            bounds=(max(0,math.floor(x1-margin)),max(0,math.floor(y1-margin)),min(image.width,math.ceil(x2+margin)),min(image.height,math.ceil(y2+margin)))
            if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:continue
            crop=image.crop(bounds)
            for scale in [1.,2.,3.,4.]:
                resized=crop.resize((max(2,round(crop.width*scale)),max(2,round(crop.height*scale))),Image.Resampling.LANCZOS)
                padded=ImageOps.expand(resized,border=4,fill='white')
                crops.append(np.asarray(padded)[:,:,::-1].copy())
                specs.append(dict(scale=scale,margin=margin,crop_box=list(bounds)))
        candidates=[]
        for language,engine in engines.items():
            # Recognition only on the identical frozen boxes; do not detect or
            # merge text regions, change graph connectivity or use a lexicon.
            rows,_=engine(crops)
            if len(rows)!=len(crops):raise RuntimeError(f'Unexpected Paddle result shape {len(rows)} vs {len(crops)}')
            for spec,row in zip(specs,rows):
                if len(row)==1 and isinstance(row[0],(list,tuple)):row=row[0]
                text,score=str(row[0]),float(row[1])
                if not math.isfinite(score) or not 0<=score<=1:raise RuntimeError('Invalid OCR confidence')
                candidates.append(dict(**spec,model=language,text=text,normalized=normalized(text),confidence=score))
        results.append(dict(key=request['key'],candidates=candidates))
        if (i+1)%32==0:
            print('RECOGNIZED',i+1,'/',len(requests),'seconds',round(time.time()-started,1),flush=True)
            (a.output/'progress.json').write_text(json.dumps(dict(status='recognizing',crops=i+1,total=len(requests))))
    hashes={str(p.relative_to(a.models)):hashlib.sha256(p.read_bytes()).hexdigest() for language in a.languages for p in (a.models/(language+'_source')).rglob('*') if p.is_file()}
    report=dict(status='completed',requests=len(requests),seconds=time.time()-started,
        backend='PaddleOCR recognition only, CPU; '+', '.join({'en':'en PP-OCRv4','latin':'latin PP-OCRv3'}[l] for l in a.languages),languages=a.languages,
        versions={name:importlib.metadata.version(name) for name in ['paddleocr','paddlepaddle','numpy']},model_sha256=hashes,
        scales=[1,2,3,4],margins=[1,3],uses_gt_text=False,uses_lexicon=False)
    (a.output/'candidates.json').write_text(json.dumps(dict(metadata=report,results=results),indent=2))
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
