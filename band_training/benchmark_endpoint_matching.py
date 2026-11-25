"""Single-thread CPU microbenchmarks on the same in-memory GT geometries."""
import argparse,json,time,pickle,platform,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .mixed_data import prepare_annotation,letterbox_transform
from .geometry import LineEndpointIndex,assemble_band_lines
from .endpoint_assignment import associate_unique_endpoints

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter()
    records=json.loads((args.dataset/'data_manifest.json').read_text());data=[]
    def load(r):return r,[json.loads((args.dataset/r[k]).read_bytes()) for k in ['annotation','band_target','line_target']]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for start in range(0,len(records),192):
            for r,(a,bt,lt) in pool.map(load,records[start:start+192]):
                ann,_,labels,quads=prepare_annotation(a,bt);scale,offset,_=letterbox_transform(a['width'],a['height'])
                lines=(np.asarray(lt['lines'])*scale+offset).tolist();boxes=[v['bbox'] for v in ann['atoms']]
                data.append(dict(sample_id=r['sample_id'],split=r['split'],annotation=r['annotation'],image=r['image'],
                    boxes=boxes,lines=lines,instances=[dict(label=l,quad=q,line=line) for l,q,line in zip(labels,quads,lines)],
                    atoms=[dict(symbol=v['symbol'],bbox=v['bbox']) for v in ann['atoms']],
                    charges=[dict(value=q['formal_charge'],bbox=q['bbox']) for q in ann['charges']],
                    reference=dict(charge_owners=[q['atom_index'] for q in ann['charges']]),labels=labels))
            if len(data)%1000<192:print('LOADED',len(data),flush=True)
    with args.cache.open('wb') as f:pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    load_seconds=time.perf_counter()-started
    methods=dict(core_forward=lambda d:LineEndpointIndex(d['boxes']).link_many(d['lines']),
                 core_exclusive=lambda d:associate_unique_endpoints(d['boxes'],d['lines'],'exclusive_box'),
                 assembly_forward=lambda d:assemble_band_lines(d['boxes'],d['instances']),
                 assembly_exclusive=lambda d:assemble_band_lines(d['boxes'],d['instances'],line_association='exclusive_box'))
    for fn in methods.values():
        for d in data[:100]:fn(d)
    results={key:[] for key in methods};times={key:[] for key in methods};rng=np.random.default_rng(716)
    for repeat in range(args.repeats):
        order=rng.permutation(len(data));keys=list(methods)
        if repeat%2:keys.reverse()
        for name in keys:
            duration=np.zeros(len(data));fn=methods[name];start=time.perf_counter()
            for i in order:
                tick=time.perf_counter_ns();fn(data[i]);duration[i]=(time.perf_counter_ns()-tick)/1e6
            wall=time.perf_counter()-start;times[name].append(duration);results[name].append(wall)
            print('BENCH',repeat,name,'seconds',round(wall,4),flush=True)
    report=dict(images=len(data),repeats=args.repeats,load_seconds=load_seconds,
        cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor()),
        available_cpus=len(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),
        scope='Input already in RAM; includes tree construction/deduplication/matching; no file IO, model inference or OCR in timed loops.',methods={})
    counts=np.array([len(d['boxes']) for d in data])
    for name in methods:
        arr=np.stack(times[name]);flat=arr.flatten();item=dict(total_seconds=results[name],median_total_seconds=float(np.median(results[name])),
            mean_ms_per_image=float(arr.mean()),median_ms_per_image=float(np.median(flat)),p95_ms_per_image=float(np.percentile(flat,95)),complexity={})
        for group,mask in [('<=25',counts<=25),('26-50',(counts>25)&(counts<=50)),('51-100',(counts>50)&(counts<=100)),('>100',counts>100)]:
            if mask.any():item['complexity'][group]=dict(images=int(mask.sum()),mean_ms=float(arr[:,mask].mean()),p95_ms=float(np.percentile(arr[:,mask],95)))
        report['methods'][name]=item
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
