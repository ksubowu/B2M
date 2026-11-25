"""Conservative line-only fine-tuning with immutable atom/charge/band weights."""
import argparse,copy,json,math,random,shutil,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset,WeightedRandomSampler
from .train import load_model,ROOT
from .continue_indigo import loader,collate,worker_init,write,sha,cpu_state
from .train_full_v3 import stratified_indices,atomic_checkpoint
from .scored_bond_data import ScoredBondDataset
from .scored_bond_model import ScoredBondDecoder,move_scored_targets,scored_bond_loss
from .train_scored_bonds import ScoredJointModel,SCHEMA,evaluate
from .line_neighborhood_loss import line_neighborhood_terms


def objective(model,images,targets):
    _,output=model(images)
    original=scored_bond_loss(output,targets);extra=line_neighborhood_terms(output['line'],targets)
    total=original['line_total']+.5*extra['neighborhood']+.5*extra['relative']+.1*extra['direction']
    return total,dict(line_total=original['line_total'],**extra)


def main():
    p=argparse.ArgumentParser();p.add_argument('--init',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=6);p.add_argument('--max-steps',type=int,default=0);p.add_argument('--tiny-steps',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--workers',type=int,default=4);p.add_argument('--val-limit',type=int,default=0)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--patience',type=int,default=2);p.add_argument('--seed',type=int,default=20261119)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);device=torch.device('cuda:0')
    source=torch.load(a.init,map_location='cpu',weights_only=False);assert source['schema']==SCHEMA;c=source['config']
    base=load_model(Path(c['base']),device).base;base.load_state_dict(source['base'])
    head=ScoredBondDecoder().to(device);head.load_state_dict(source['decoder']);model=ScoredJointModel(base,head).to(device)
    model.requires_grad_(False);head.line_head.requires_grad_(True);model.eval();head.line_head.train()
    train=ScoredBondDataset(c['dataset'],'train',augment=True);val=ScoredBondDataset(c['dataset'],'val')
    assets=json.loads((ROOT/'runs/full_v3_training_audit_20261116/assets.json').read_text());lookup={r['sample_id']:r for r in assets}
    source_id=lambda r:r['sample_id'].split('::repeat_')[0]
    assert not ({source_id(r) for r in train.records}&{source_id(r) for r in val.records})
    weights=[]
    for r in train.records:
        meta=lookup[source_id(r)];weights.append(1.+.5*(meta['bonds']>=40)+.5*any(int(k) in [4,5] and v for k,v in meta['bond_classes'].items()))
    sampler=WeightedRandomSampler(weights,len(train),replacement=True,generator=torch.Generator().manual_seed(a.seed))
    training=DataLoader(train,batch_size=a.batch_size,sampler=sampler,num_workers=a.workers,pin_memory=True,collate_fn=collate,worker_init_fn=worker_init,persistent_workers=a.workers>0)
    vi=stratified_indices(val,assets,a.val_limit,a.seed);validation=loader(Subset(val,vi),a.batch_size,a.workers)
    for split in ['train','val']:shutil.copyfile(Path(c['dataset'])/(split+'.csv'),a.output/(split+'.csv'))
    config=dict(c,**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},schema=SCHEMA,
                line_threshold=.25,band_threshold=.2,refine_line_peaks=True,stereo_from_band=True,
                training_strategy='Only line_head trainable; every atom/charge/shared/band parameter and buffer frozen.',
                neighborhood_loss_weights=dict(neighborhood=.5,relative=.5,direction=.1),
                sampling='Existing train.csv including enriched charges; multiply complex (>=40 bonds) by +0.5 and directed-bond images by +0.5. No validation examples used.',
                manifest_sha256={s:sha(Path(c['dataset'])/(s+'.csv')) for s in ['train','val']},init_sha256=sha(a.init))
    config.update(train_rows=len(train),unique_train=len({source_id(r) for r in train.records}),validation_images=len(val),evaluated_val_images=len(vi))
    write(a.output/'config.json',config);write(a.output/'validation_sample_ids.json',[val.records[i]['sample_id'] for i in vi])
    snapshot=a.output/'source';snapshot.mkdir()
    for name in ['finetune_line_neighborhood.py','line_neighborhood_loss.py','scored_bond_model.py','scored_bond_postprocess.py','train_scored_bonds.py']:
        shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    args=argparse.Namespace(line_threshold=.25,band_threshold=.2,refine_line_peaks=True,stereo_from_band=True)
    started=time.time();write(a.output/'progress.json',dict(status='baseline_validation',step=0,validation_images=len(vi)))
    baseline=evaluate(model,validation,device,args,a.output,'baseline')
    same_manifest=c.get('manifest_sha256',{}).get('val')==config['manifest_sha256']['val']
    same_postprocess=all(c.get(k,False)==getattr(args,k) for k in ['line_threshold','band_threshold','refine_line_peaks','stereo_from_band'])
    expected=source.get('metrics',{}).get('counts',{}).get('full_graph_exact')
    if not a.val_limit and same_manifest and same_postprocess and expected is not None:
        assert baseline['counts']['full_graph_exact']==expected,'Frozen baseline changed; stop before optimization'
    steps=0;max_grad=0.;history=[];tiny=None;original_line=copy.deepcopy(head.line_head.state_dict())
    if a.tiny_steps:
        images,metas=next(iter(training));images=images.to(device);targets=move_scored_targets([m['scored_bonds'] for m in metas],device)
        opt=torch.optim.AdamW(head.line_head.parameters(),lr=a.lr,weight_decay=1e-4)
        with torch.no_grad():initial=float(objective(model,images,targets)[0])
        for _ in range(a.tiny_steps):
            opt.zero_grad(set_to_none=True);loss,_=objective(model,images,targets);loss.backward();torch.nn.utils.clip_grad_norm_(head.line_head.parameters(),5,error_if_nonfinite=True);opt.step()
        with torch.no_grad():final=float(objective(model,images,targets)[0])
        tiny=dict(initial=initial,final=final,steps=a.tiny_steps,passed=math.isfinite(final) and final<initial*.99)
        write(a.output/'tiny_overfit.json',tiny);assert tiny['passed'],'Line-only objective did not learn';head.line_head.load_state_dict(original_line)
    opt=torch.optim.AdamW(head.line_head.parameters(),lr=a.lr,weight_decay=1e-4)
    total=a.max_steps or len(training)*a.epochs;warmup=min(100,max(1,total//10))
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda i:min(1.,(i+1)/warmup)*(.1+.9*.5*(1+math.cos(math.pi*min(i,total)/total))))
    best=(baseline['counts']['full_graph_exact'],baseline['typed_bond']['f1']);stale=0;reason='epochs_completed'
    def save(path,epoch,metrics):
        atomic_checkpoint(dict(schema=SCHEMA,base=cpu_state(base),decoder=cpu_state(head),step=steps,next_epoch=epoch,config=config,
            metrics=metrics,best_score=list(best),optimizer=opt.state_dict(),scheduler=schedule.state_dict()),path)
    save(a.output/'best_joint.pt',0,baseline)
    with (a.output/'history.jsonl').open('w') as log:
        for epoch in range(a.epochs):
            model.eval();head.line_head.train()
            for images,metas in training:
                images=images.to(device);targets=move_scored_targets([m['scored_bonds'] for m in metas],device)
                opt.zero_grad(set_to_none=True);loss,terms=objective(model,images,targets)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(head.line_head.parameters(),5,error_if_nonfinite=True))
                assert grad>0;max_grad=max(max_grad,grad);opt.step();schedule.step();steps+=1
                row=dict(step=steps,epoch=epoch+1,loss=float(loss.detach()),gradient_norm=grad,**{k:float(v.detach()) for k,v in terms.items()})
                history.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if steps==1 or steps%20==0:
                    write(a.output/'progress.json',dict(status='training',step=steps,epoch=epoch+1,steps_per_epoch=len(training),total_epochs=a.epochs,seconds=time.time()-started))
                    print('TRAIN',json.dumps(row),flush=True)
                if a.max_steps and steps>=a.max_steps:break
            write(a.output/'progress.json',dict(status='epoch_validation',epoch=epoch+1,step=steps))
            metrics=evaluate(model,validation,device,args,a.output,f'epoch_{epoch+1:02d}')
            assert all(torch.equal(v.cpu(),source['base'][k]) for k,v in base.state_dict().items())
            assert all(torch.equal(v.cpu(),source['decoder'][k]) for k,v in head.state_dict().items() if not k.startswith('line_head.'))
            score=(metrics['counts']['full_graph_exact'],metrics['typed_bond']['f1'])
            if score>best:best=score;stale=0;save(a.output/'best_joint.pt',epoch+1,metrics)
            else:stale+=1
            save(a.output/'last_joint.pt',epoch+1,metrics)
            if a.max_steps and steps>=a.max_steps:reason='smoke_steps_completed';break
            if stale>=a.patience:reason='early_stop_no_improvement';break
    passed=(metrics['atom_f1']==baseline['atom_f1'] and metrics['full_graph_exact']>=baseline['full_graph_exact']-.01 and max_grad>0)
    result=dict(status=('passed' if passed else 'failed') if a.max_steps else 'completed',reason=reason,steps=steps,seconds=time.time()-started,
        baseline=baseline,final=metrics,best_full_exact=best[0],tiny_overfit=tiny,immutable_weights_verified=True,max_gradient_norm=max_grad,
        note='Best checkpoint includes unchanged initialization as fallback. Validation-set development results, not independent test accuracy.')
    write(a.output/'results.json',result);write(a.output/'progress.json',dict(status=result['status'],step=steps,reason=reason));print('DONE',json.dumps(result),flush=True)
    if a.max_steps and not passed:raise RuntimeError('Smoke guard failed')


if __name__=='__main__':main()
