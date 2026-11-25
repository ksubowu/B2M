"""Compare fixed KD-tree nearest-center linking with the original geometry rule.

This is a postprocessing experiment on frozen predictions, not a retraining or
replacement of the original pilot results. Timings include tree construction
once per image and batch endpoint queries; model inference/NMS/IO are excluded.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .geometry import aggregate,edge_metrics,link_many,suppress,atom_mapping
from .train import decode_atoms,load_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--repeats',type=int,default=7);args=p.parse_args()
    torch.set_num_threads(2)
    config=json.loads((args.run/'config.json').read_text());results=json.loads((args.run/'results.json').read_text())
    records=torch.load(args.run/'feature_cache.pt',map_location='cpu',weights_only=False)
    saved=json.loads((args.run/'test_predictions.json').read_text())
    cases=[];prediction_cases=[];baseline=[];adapted=[]
    for r in records:
        truth=r['ann'];boxes=[a['bbox'] for a in truth['atoms']]
        bands=[dict(label=l,quad=q,score=1.) for l,q in zip(r['band']['labels'],r['band']['quads'])]
        cases.append(dict(image=r['row']['image'],split=r['row']['split'],bands=bands,boxes=boxes,truth=truth))
        if r['row']['split']=='test':
            ps=suppress([p for p in saved[r['row']['image']] if p['score']>=results['threshold']])
            prediction_cases.append(dict(image=r['row']['image'],split='test',bands=ps,boxes=boxes,truth=truth))
    # Obtain final atom-head detections from frozen cached features. Band predictions stay fixed.
    decoder=load_model(Path(config['base']),args.device).base.box_decoder.eval()
    decoder.load_state_dict(torch.load(args.run/'atom_decoder.pth',map_location=args.device,weights_only=True))
    with torch.no_grad():
        for r,case in zip((r for r in records if r['row']['split']=='test'),prediction_cases):
            ap=decode_atoms(decoder([v[None].float().to(args.device) for v in r['features']]))
            atoms=[a for a in ap if a['label']<13];mapping,_=atom_mapping(atoms,r['ann'])
            adapted.append({**case,'boxes':[a['bbox'] for a in atoms],'mapping':mapping})
    groups={**{f'true_bands_true_boxes_{split}':[c for c in cases if c['split']==split] for split in ['train','calibration','test']},
            'predicted_bands_true_boxes_test':prediction_cases,'predicted_bands_predicted_boxes_test':adapted}
    report={'policy':'Postprocessing comparison on fixed annotations/predictions. Original results.json and trained weights unchanged.',
            'centers':'bbox x+w/2, y+h/2; never annotation atom.center',
            'kdtree':'One cKDTree per image; batch k=1 queries on ordered band ends; reject same-atom/degenerate connections; no geometric rescoring.',
            'metrics':{},'differences':{},'timings':{}}
    for name,items in groups.items():
        report['metrics'][name]={}
        for method in ['geometry','kdtree']:
            rows=[edge_metrics(c['bands'],c['truth'],c['boxes'],c.get('mapping'),method) for c in items]
            report['metrics'][name][method]=aggregate(rows)
        differences=[]
        for c in items:
            quads=[p['quad'] for p in c['bands']]
            old=link_many(quads,c['boxes'],'geometry');new=link_many(quads,c['boxes'],'kdtree')
            changed=[dict(band=i,geometry=a,kdtree=b) for i,(a,b) in enumerate(zip(old,new)) if a!=b]
            if changed:differences.append(dict(image=c['image'],changes=changed))
        report['differences'][name]=differences
    # Warmup; then repeated whole-corpus timings in alternating order to reduce order bias.
    for method in ['geometry','kdtree']:
        for c in cases[:8]:link_many([p['quad'] for p in c['bands']],c['boxes'],method)
    times={'geometry':[],'kdtree':[]}
    for repeat in range(args.repeats):
        for method in (['geometry','kdtree'] if repeat%2==0 else ['kdtree','geometry']):
            start=time.perf_counter()
            for c in cases:link_many([p['quad'] for p in c['bands']],c['boxes'],method)
            times[method].append(time.perf_counter()-start)
    for method,seconds in times.items():
        report['timings'][method]={'all_images_seconds_median':float(np.median(seconds)),
                                  'milliseconds_per_image_median':float(np.median(seconds)*1000/len(cases)),
                                  'repeats':args.repeats,'images':len(cases),'bands':sum(len(c['bands']) for c in cases)}
    report['timings']['speedup']=report['timings']['geometry']['all_images_seconds_median']/report['timings']['kdtree']['all_images_seconds_median']
    dest=args.run/'kdtree_comparison.json';dest.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='differences'},indent=2),flush=True)
    print('Saved',dest)

if __name__=='__main__':main()
