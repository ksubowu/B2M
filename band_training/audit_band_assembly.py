"""Audit GT geometry without passing GT identities into assembly; optional reevaluation."""
import argparse,csv,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from PIL import Image,ImageDraw
from .mixed_data import MixedIndigoDataset,prepare_annotation
from .geometry import link_many,edge_key,aggregate
from .continue_indigo import write,sha,JointModel,loader,evaluate,load_model
from .model import BandDecoder


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run',type=Path);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);report={};failures=[];started=time.time()
    for split in ['val','train']:
        ds=MixedIndigoDataset(args.dataset,split);seen=set();rows={m:[] for m in ['kdtree','band_directional']}
        for ri,r in enumerate(ds.records):
            if r['annotation'] in seen:continue
            seen.add(r['annotation']);a=json.loads((args.dataset/r['annotation']).read_text());t=json.loads((args.dataset/r['band_target']).read_text())
            ann,_,labels,quads=prepare_annotation(a,t);boxes=[v['bbox'] for v in ann['atoms']]
            truth=Counter(edge_key(b['begin'],b['end'],b['category_id']-1) for b in ann['bonds'])
            for method in rows:
                pairs=link_many(quads,boxes,method=method);edges=Counter(edge_key(*pair,label) for pair,label in zip(pairs,labels) if pair is not None)
                tp=sum((edges&truth).values());unlinked=sum(pair is None for pair in pairs)
                rows[method].append(dict(tp=tp,fp=len(labels)-tp,fn=len(labels)-tp,unlinked=unlinked,exact=int(edges==truth and not unlinked),gt=len(labels)))
                if method=='band_directional':
                    for i,(pair,bond,label) in enumerate(zip(pairs,ann['bonds'],labels)):
                        correct=pair is not None and edge_key(*pair,label)==edge_key(bond['begin'],bond['end'],label)
                        if not correct:
                            failure=dict(split=split,sample_id=r['sample_id'],band_index=i,label=label,predicted=pair,
                                         expected=[bond['begin'],bond['end']],quad=quads[i],record=r)
                            failures.append(failure)
            if ri%2000==0:print('ORACLE',split,ri,'unique',len(seen),flush=True)
        report[split]=dict(unique_images=len(seen),sampling_rows=len(ds),manifest_sha256=sha(ds.manifest),
                           methods={m:aggregate(v) for m,v in rows.items()})
        write(args.output/'oracle_report.json',report);write(args.output/'oracle_failures.json',failures)
        print('ORACLE_RESULT',split,json.dumps(report[split]),flush=True)
    # Review pictures contain expected identities ONLY as visualization labels.
    # The assembly functions above received exclusively boxes and quad coordinates.
    for number,f in enumerate([v for v in failures if v['split']=='val'][:12]):
        r=f['record'];ds=MixedIndigoDataset(args.dataset,'val');idx=next(i for i,x in enumerate(ds.records) if x['sample_id']==r['sample_id'])
        tensor,m=ds[idx];base=Image.fromarray((tensor.permute(1,2,0).numpy()*255).round().astype('uint8'))
        im=base.copy();d=ImageDraw.Draw(im);q=f['quad'];d.line([tuple(v) for v in q]+[tuple(q[0])],fill='red',width=2)
        ids=set(f['expected']+(list(f['predicted']) if f['predicted'] else []))
        for i in ids:
            atom=m['ann']['atoms'][i];x,y,w,h=atom['bbox'];d.rectangle([x,y,x+w,y+h],outline='blue',width=1);d.text((x,y-12),str(i)+' '+atom['symbol'],fill='blue')
        canvas=Image.new('RGB',(1280,680),'white');canvas.paste(base,(0,40));canvas.paste(im,(640,40))
        ImageDraw.Draw(canvas).text((5,5),f['sample_id']+' | band '+str(f['band_index'])+' | red=band; blue=atom boxes',fill='black')
        canvas.save(args.output/f'oracle_failure_{number:02d}.png')
    if args.run:
        torch.manual_seed(20261114);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        config=json.loads((args.run/'config.json').read_text());device=torch.device('cuda:0')
        state=torch.load(args.run/'best_joint.pt',map_location='cpu',weights_only=False)
        base=load_model(Path(config['base']),device).base;base.load_state_dict(state['base'],strict=True)
        band=BandDecoder().to(device);band.load_state_dict(state['band'],strict=True)
        model=JointModel(base,band).to(device).eval();ds=MixedIndigoDataset(args.dataset,'val')
        metrics=evaluate(model,loader(ds,8,4),device,state['threshold'],args.output,'directional',association='band_directional')
        old=json.loads((args.run/'results.json').read_text())['best_metrics']
        assert old['atom_f1']==metrics['atom_f1'] and old['charge_f1']==metrics['charge_f1'],'Detection outputs changed; do not conflate comparisons'
        write(args.output/'comparison.json',dict(checkpoint=str(args.run/'best_joint.pt'),checkpoint_sha256=sha(args.run/'best_joint.pt'),
                 old={k:old[k] for k in ['atom_f1','charge_f1','bond_graph','full_graph_exact']},
                 new={k:metrics[k] for k in ['atom_f1','charge_f1','bond_graph','full_graph_exact']},seconds=time.time()-started))

if __name__=='__main__':main()
