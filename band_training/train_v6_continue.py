"""Warm-start all scored heads on v6, uniform sampling and full Paddle validation."""
import argparse,json,math,os,random,shutil,subprocess,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from .checkpoint_runtime import load_joint_base
from .train import ROOT,load_model
from .continue_indigo import loader,write,sha,cpu_state
from .continuation_data import ContinuationDataset
from .scored_bond_model import ScoredBondDecoder,move_scored_targets,scored_bond_loss
from .train_scored_bonds import ScoredJointModel,SCHEMA
from .charge_loss import ChargeAwareCriterion
from .line_neighborhood_loss import line_neighborhood_terms
from .train_full_v3 import atomic_checkpoint
from .summarize_full_paddle_metrics import summarize_split
from engine.core import YAMLConfig
from .plot_v6_training import render_monitor


def select_smoke(records,metadata,limit,seed,include_all_charged=False):
    if not limit or limit>=len(records):return list(range(len(records)))
    info={m['sample_id']:m for m in metadata};selected=[]
    def add(i):
        if i not in selected and len(selected)<limit:selected.append(i)
    for i in sorted(range(len(records)),key=lambda i:-(info[records[i]['sample_id']]['atoms']+info[records[i]['sample_id']]['charges']))[:2]:add(i)
    for label in [8,9,10,11,12,0]:
        eligible=[i for i,r in enumerate(records) if info[r['sample_id']]['atom_classes'].get(str(label),0)]
        if eligible:add(eligible[len(eligible)//2])
    for label in [15,16]:
        eligible=[i for i,r in enumerate(records) if info[r['sample_id']]['charge_classes'].get(str(label),0)]
        if eligible:add(eligible[len(eligible)//2])
    for label in [2,4,5]:
        eligible=[i for i,r in enumerate(records) if info[r['sample_id']]['bond_classes'].get(str(label),0)]
        if eligible:add(eligible[len(eligible)//2])
    if include_all_charged:
        charged=[i for i,r in enumerate(records) if info[r['sample_id']]['charges']]
        if len(charged)<=limit//2:
            for i in charged:add(i)
    order=list(range(len(records)));random.Random(seed).shuffle(order)
    for i in order:add(i)
    return selected


def child(command,log):
    with log.open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())


def validation(model,config,out,tag,step,epoch):
    model.eval();candidate=out/'validation_candidate.pt'
    atomic_checkpoint(dict(schema=SCHEMA,base=cpu_state(model.base),decoder=cpu_state(model.decoder),step=step,next_epoch=epoch,config=config,metrics={}),candidate)
    phase=out/'validation'/tag;phase.mkdir(parents=True,exist_ok=False)
    main=sys.executable;dataset=config['dataset'];shared=out/'reference_verified'
    if not (shared/'report.json').exists():
        child([main,'-m','band_training.audit_graph_references','--dataset',dataset,'--manifest',config['val_manifest'],'--output',str(shared)],out/'reference_audit.log')
        assert json.loads((shared/'report.json').read_text())['status']=='passed'
    (phase/'reference_verified').symlink_to(os.path.relpath(shared,phase),target_is_directory=True)
    child([main,'-m','band_training.cache_final_predictions','--checkpoint',str(candidate),'--output',str(phase/'prediction_cache'),'--workers','4'],phase/'predict.log')
    child([main,'-m','band_training.prepare_paddle_ocr_requests','--cache',str(phase/'prediction_cache/predictions.jsonl.gz'),
           '--dataset',dataset,'--include-gt-diagnostics','--output',str(phase/'requests.json')],phase/'requests.log')
    child([os.environ.get('B2M_PADDLE_PYTHON',str(ROOT/'.venvs/paddleocr311/bin/python')),str(ROOT/'band_training/paddle_multiscale_worker.py'),'--requests',str(phase/'requests.json'),
           '--output',str(phase/'recognition'),'--models',os.environ.get('B2M_PADDLE_MODELS',str(ROOT/'webapp_data/paddleocr_models')),'--languages','en'],phase/'ocr.log')
    child([main,'-m','band_training.evaluate_graph_ocr','--cache',str(phase/'prediction_cache/predictions.jsonl.gz'),'--dataset',dataset,
           '--output',str(phase/'graph_ocr'),'--paddle-results',str(phase/'recognition/candidates.json'),'--paddle-policy','en_consensus','--mcs-examples','0'],phase/'graph.log')
    child([main,'-m','band_training.rescore_functional_groups','--evaluation',str(phase/'graph_ocr'),'--references',str(shared/'samples.json')],phase/'functional.log')
    detail,samples=summarize_split(phase);c=detail['counts'];objects=detail['objects'];whole=detail['whole_images']
    rare=[r['f1'] for r in detail['per_class'] if r['component']=='atom' and r['label'] in [5,8,9,10,11,12] and r['ground_truth'] and r['f1'] is not None]
    metrics=dict(images=c['images'],counts=dict(c,full_graph_exact=c['legacy_full_exact']),
        atom_f1=objects['atom']['f1'],charge_f1=objects['charge']['f1'],
        typed_bond=dict(precision=objects['bond']['precision'],recall=objects['bond']['recall'],f1=objects['bond']['f1'],
                        tp=c['bond_tp'],fp=c['bond_fp'],fn=c['bond_fn'],exact=c['bond_exact'],graph_exact=whole['bond_exact']['rate']),
        full_graph_exact=whole['legacy_full_exact']['rate'],graph_star_exact=whole['graph_star_exact']['rate'],graph_ocr_exact=whole['graph_ocr_exact']['rate'],
        graph_star_correct=whole['graph_star_exact']['correct'],graph_ocr_correct=whole['graph_ocr_exact']['correct'],
        rare_atom_macro_f1=float(np.mean(rare)) if rare else None,objects=objects,whole_images=whole,
        charge_subsets=detail['charge_subsets'],per_class=detail['per_class'],ocr_backend=detail['ocr_backend'])
    write(phase/'component_report.json',detail);write(out/(tag+'_metrics.json'),metrics)
    print('VALIDATION',tag,json.dumps({k:metrics[k] for k in ['images','atom_f1','charge_f1','full_graph_exact','graph_star_exact','graph_ocr_exact','rare_atom_macro_f1']}),flush=True)
    return metrics


def key(metrics):return (metrics['graph_ocr_correct'],metrics['graph_star_correct'],metrics['counts']['full_graph_exact'],metrics['typed_bond']['f1'],metrics['rare_atom_macro_f1'] or 0.)


def acceptable(metrics,baseline):
    return metrics['atom_f1']>=baseline['atom_f1']-.003 and metrics['charge_f1']>=baseline['charge_f1']-.03 and metrics['typed_bond']['f1']>=baseline['typed_bond']['f1']-.003


def configure_box_training(base,freeze_boxes=False):
    base.box_decoder.requires_grad_(not freeze_boxes)
    if freeze_boxes:base.box_decoder.eval()


def extended_lr_factor(step, start_step, end_step, start_factor):
    progress=max(0.,min(1.,(step-start_step)/max(1,end_step-start_step)))
    return start_factor*(.1+.9*.5*(1+math.cos(math.pi*progress)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True);p.add_argument('--init',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=6);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--accumulation',type=int,default=1);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--max-steps',type=int,default=0);p.add_argument('--train-limit',type=int,default=0);p.add_argument('--val-limit',type=int,default=0)
    p.add_argument('--freeze-feature-steps',type=int,default=500);p.add_argument('--warmup-steps',type=int,default=300);p.add_argument('--patience',type=int,default=2)
    p.add_argument('--checkpoint-every',type=int,default=2000);p.add_argument('--memory-fraction',type=float,default=.4);p.add_argument('--seed',type=int,default=20261121);p.add_argument('--resume',type=Path)
    p.add_argument('--freeze-boxes',action='store_true',help='Freeze atom/charge box parameters and use eval mode during bond-only training')
    p.add_argument('--training-manifest',type=Path,help='Audited subset of effective preflight training records')
    p.add_argument('--box-lr',type=float,default=1e-5);p.add_argument('--shared-bond-lr',type=float,default=1e-5)
    p.add_argument('--line-lr',type=float,default=3e-5);p.add_argument('--band-lr',type=float,default=3e-5)
    p.add_argument('--extend-training',action='store_true',help='Extend a completed run, retaining optimizer and current LR without a restart')
    a=p.parse_args();a.dataset=a.dataset.resolve();a.preflight=a.preflight.resolve();a.init=a.init.resolve();a.output=a.output.resolve()
    if a.extend_training and not a.resume:p.error('--extend-training requires --resume')
    audit=json.loads((a.preflight/'report.json').read_text());assert audit['status']=='passed'
    assert sha(a.dataset/'train.json')==audit['source_hashes']['train'] and sha(a.dataset/'val.json')==audit['source_hashes']['val']
    a.output.mkdir(parents=True,exist_ok=bool(a.resume));torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(a.memory_fraction,device)
    source=torch.load(a.init,map_location='cpu',weights_only=False);assert source['schema']==SCHEMA
    train=ContinuationDataset(a.dataset,'train',manifest=a.training_manifest or a.preflight/'train.json',augment=True);val=ContinuationDataset(a.dataset,'val',manifest=a.preflight/'val.json')
    if a.training_manifest:
        allowed={r['sample_id']:r for r in json.loads((a.preflight/'train.json').read_text())}
        assert len({r['sample_id'] for r in train.records})==len(train.records)
        assert all(r==allowed.get(r['sample_id']) for r in train.records),'Subset must contain unchanged audited training records'
    metadata=json.loads((a.preflight/'assets.json').read_text());ti=select_smoke(train.records,metadata,a.train_limit,a.seed);vi=select_smoke(val.records,metadata,a.val_limit,a.seed+1,include_all_charged=True)
    if not a.resume:
        write(a.output/'train.json',[train.records[i] for i in ti]);write(a.output/'val.json',[val.records[i] for i in vi])
    train.records=json.loads((a.output/'train.json').read_text());val.records=json.loads((a.output/'val.json').read_text())
    assert not ({r['sample_id'] for r in train.records}&{r['sample_id'] for r in val.records})
    assert not ({r['group'] for r in train.records}&{r['group'] for r in val.records})
    base=load_joint_base(source,device);decoder=ScoredBondDecoder().to(device);decoder.load_state_dict(source['decoder'])
    model=ScoredJointModel(base,decoder).to(device);base.requires_grad_(True);base.line_decoder.requires_grad_(False);configure_box_training(base,a.freeze_boxes)
    shared=list(decoder.project.parameters())+list(decoder.image.parameters())+list(decoder.body.parameters())
    groups=[dict(params=base.backbone.parameters(),lr=1e-6,name='backbone'),dict(params=base.encoder.parameters(),lr=2e-6,name='encoder'),
            dict(params=base.box_decoder.parameters(),lr=a.box_lr,name='boxes'),dict(params=shared,lr=a.shared_bond_lr,name='shared_bond'),
            dict(params=decoder.line_head.parameters(),lr=a.line_lr,name='line_head'),dict(params=decoder.band_head.parameters(),lr=a.band_lr,name='band_head')]
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4);batches=math.ceil(len(train)/a.batch_size);updates_per_epoch=math.ceil(batches/a.accumulation)
    total=a.max_steps or updates_per_epoch*a.epochs;warmup=min(a.warmup_steps,max(1,total//10))
    recovered=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
    continuation=None
    if a.extend_training:
        assert a.epochs>recovered['resume_epoch']
        continuation=(recovered['config'].get('lr_continuation') if recovered['config']['epochs']==a.epochs else None)
        if continuation is None:
            continuation=dict(start_step=recovered['step'],end_step=total,start_factor=recovered['optimizer']['param_groups'][0]['lr']/recovered['optimizer']['param_groups'][0]['initial_lr'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,(lambda step:extended_lr_factor(step,**continuation)) if continuation else lambda step:min(1.,(step+1)/warmup)*(.1+.9*.5*(1+math.cos(math.pi*min(step,total)/total))))
    cfg=YAMLConfig(str(ROOT/'config/train_mp123_full.yml'));criterion=ChargeAwareCriterion(cfg.criterion.box_criterion.to(device).train())
    config=dict(source['config'],**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    config.update(schema=SCHEMA,dataset=str(a.dataset),base=source['config']['base'],train_manifest=str(a.output/'train.json'),val_manifest=str(a.output/'val.json'),
        train_rows=len(train),unique_train=len(train),validation_images=len(val),evaluated_val_images=len(val),batch_size=a.batch_size,
        line_threshold=.25,band_threshold=.2,refine_line_peaks=True,stereo_from_band=True,
        manifest_sha256={s:sha(a.output/(s+'.json')) for s in ['train','val']},init_sha256=sha(a.init),source_manifest_sha256=audit['source_hashes'],
        sampling='Uniform shuffle of unique audited selected records; each image once per epoch. No repeated entries or extra weighted sampling.',
        mini_hard_exclusions=dict(known=audit['known_mini_excluded'],additional=audit['extra_mini_excluded']),
        training_strategy=('Frozen backbone/encoder throughout; box/charge and shared bond/line/band heads trainable; BN statistics frozen.' if a.freeze_feature_steps>=total else 'All heads trainable; backbone/encoder warmed frozen then low LR; BN statistics frozen.'),
        optimizer_groups={g['name']:g['initial_lr'] for g in optimizer.param_groups},losses='ChargeAware box criterion + 2*(independent scored line/band + .5 neighborhood + .5 relative ordered endpoints + .1 direction)',
        selection='Paddle en multi-scale consensus exact graph first; atom/charge/bond F1 regression guards; initialization retained as fallback.',
        target_audit=str(a.preflight/'report.json'))
    config.pop('freeze_base_epochs',None)
    config.update(early_stopping_enabled=a.patience>0,lr_continuation=continuation)
    if a.freeze_boxes:
        config['training_strategy']+=' Box decoder parameters frozen and eval mode enforced; bond-only training.'
        config['losses']='No box/charge optimization loss; 2*(independent scored line/band + .5 neighborhood + .5 relative ordered endpoints + .1 direction)'
    write(a.output/'config.json',config);write(a.output/'validation_sample_ids.json',[r['sample_id'] for r in val.records])
    snap=a.output/'source';snap.mkdir(exist_ok=True)
    for name in ['train_v6_continue.py','continuation_data.py','scored_bond_model.py','scored_bond_postprocess.py','line_neighborhood_loss.py','charge_loss.py','cache_final_predictions.py','dataset_runtime.py','plot_v6_training.py','checkpoint_runtime.py']:
        shutil.copyfile(Path(__file__).with_name(name),snap/name)
    start=time.time();step=0;start_epoch=0;start_batch=0;stale=0;history=[];max_grad=0.;best_metrics=None;baseline=None;best_key=None
    def save(path,epoch,next_batch,metrics,complete=False):
        state=dict(schema=SCHEMA,base=cpu_state(base),decoder=cpu_state(decoder),step=step,next_epoch=epoch+int(complete),
            resume_epoch=epoch+int(complete),resume_batch=0 if complete else next_batch,config=config,metrics=metrics or {},
            best_metrics=best_metrics,best_key=best_key,baseline=baseline,stale=stale,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()))
        atomic_checkpoint(state,path)
    if a.resume:
        for field in ['manifest_sha256','batch_size','accumulation']+([] if a.extend_training else ['epochs']):assert recovered['config'][field]==config[field]
        base.load_state_dict(recovered['base']);decoder.load_state_dict(recovered['decoder']);optimizer.load_state_dict(recovered['optimizer']);scheduler.load_state_dict(recovered['scheduler'])
        step=recovered['step'];start_epoch=recovered['resume_epoch'];start_batch=recovered['resume_batch'];baseline=recovered['baseline'];best_metrics=recovered['best_metrics'];best_key=tuple(recovered['best_key']);stale=recovered['stale']
        rng=recovered['rng'];torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
    else:
        write(a.output/'progress.json',dict(status='baseline_validation',step=0,validation_images=len(val)))
        baseline=validation(model,config,a.output,'baseline',0,0);best_metrics=baseline;best_key=key(baseline);save(a.output/'best_joint.pt',0,0,baseline)
    render_monitor(a.output)
    gradients_seen=Counter();reason='epochs_completed';epoch_metrics=[]
    with (a.output/'history.jsonl').open('a') as log:
        for epoch in range(start_epoch,a.epochs):
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(a.seed+epoch)).tolist()
            offset=start_batch*a.batch_size if epoch==start_epoch else 0
            training=loader(Subset(train,order[offset:]),a.batch_size,a.workers);model.train();base.backbone.eval();base.encoder.eval();base.line_decoder.eval();configure_box_training(base,a.freeze_boxes)
            optimizer.zero_grad(set_to_none=True);window=[];micro=0;epoch_start=time.time();seen=offset
            for local_batch,(images,metas) in enumerate(training):
                tick=time.time();absolute_batch=offset//a.batch_size+local_batch;active=step>=a.freeze_feature_steps
                base.backbone.requires_grad_(active);base.encoder.requires_grad_(active)
                images=images.to(device,non_blocking=True);bt=[{k:v.to(device,non_blocking=True) for k,v in m['boxes'].items()} for m in metas]
                targets=move_scored_targets([m['scored_bonds'] for m in metas],device);boxes,output=model(images,bt)
                bond=scored_bond_loss(output,targets);extra=line_neighborhood_terms(output['line'],targets)
                # Eval-mode box decoder omits training auxiliary outputs. A
                # frozen box branch has no supervised optimization objective.
                box_terms={} if a.freeze_boxes else criterion(boxes,bt)
                box_loss=sum(box_terms.values(),torch.zeros((),device=device))
                loss=box_loss+2*(bond['total']+.5*extra['neighborhood']+.5*extra['relative']+.1*extra['direction'])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite continuation loss')
                (loss/a.accumulation).backward();micro+=1;seen+=len(metas)
                last=absolute_batch+1==batches
                if micro%a.accumulation and not last:continue
                # A group's first tensor may legitimately have zero gradient
                # (for example an unused query embedding). Check the whole
                # group and synchronize only once per group.
                for group in optimizer.param_groups:
                    norms=[p.grad.detach().abs().max() for p in group['params'] if p.grad is not None]
                    if norms:gradients_seen[group['name']]+=int(bool(torch.stack(norms).max()>0))
                grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));max_grad=max(max_grad,grad)
                assert grad>0;optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
                seconds=time.time()-tick;window.append(seconds)
                if len(window)>100:window.pop(0)
                row=dict(step=step,epoch=epoch+1,batch=absolute_batch+1,images_seen=seen,features_trainable=active,
                    loss=float(loss.detach()),box_loss=float(box_loss.detach()),line_loss=float(bond['line_total'].detach()),band_loss=float(bond['band_total'].detach()),
                    neighborhood=float(extra['neighborhood'].detach()),gradient_norm=grad,seconds=seconds,peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3)
                row.update(box_supervision_active=not a.freeze_boxes,learning_rates={g['name']:g['lr'] for g in optimizer.param_groups})
                if box_terms:row['charge_aux_loss']=float(sum(v for k,v in box_terms.items() if k.startswith('loss_charge_')).detach())
                history.append(row)
                if step==1 or step%20==0:
                    log.write(json.dumps(row)+'\n');log.flush();print('TRAIN',json.dumps(row),flush=True)
                    write(a.output/'progress.json',dict(status='training',**row,total_epochs=a.epochs,total_steps=total,steps_per_epoch=updates_per_epoch,
                        train_images=len(train),validation_images=len(val),mean_step_seconds=float(np.mean(window)),epoch_seconds=time.time()-epoch_start,total_seconds=time.time()-start))
                if step%200==0:render_monitor(a.output)
                if a.checkpoint_every and step%a.checkpoint_every==0:save(a.output/'recovery.pt',epoch,absolute_batch+1,best_metrics)
                if a.max_steps and step>=a.max_steps:break
            optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            write(a.output/'progress.json',dict(status='epoch_validation',step=step,epoch=epoch+1,validation_images=len(val)))
            metrics=validation(model,config,a.output,f'epoch_{epoch+1:02d}',step,epoch+1);epoch_metrics.append(metrics)
            allowed=acceptable(metrics,baseline);candidate_key=key(metrics)
            if allowed and candidate_key>best_key:best_key=candidate_key;best_metrics=metrics;stale=0;save(a.output/'best_joint.pt',epoch,0,metrics,complete=True)
            else:stale+=1
            save(a.output/'last_joint.pt',epoch,0,metrics,complete=True)
            save(a.output/f'epoch_{epoch+1:02d}.pt',epoch,0,metrics,complete=True)
            write(a.output/'selection.json',dict(epoch=epoch+1,passes_regression_guards=allowed,stale_epochs=stale,best_key=best_key,best_metrics=best_metrics,current_metrics=metrics))
            render_monitor(a.output)
            start_batch=0
            if a.max_steps and step>=a.max_steps:reason='smoke_steps_completed';break
            if a.patience>0 and stale>=a.patience:reason='early_stop_no_improvement';break
    changed={}
    for name,module,old,prefix in [('backbone',base.backbone,source['base'],'backbone.'),('encoder',base.encoder,source['base'],'encoder.'),('boxes',base.box_decoder,source['base'],'box_decoder.'),
                                   ('line_head',decoder.line_head,source['decoder'],'line_head.'),('band_head',decoder.band_head,source['decoder'],'band_head.')]:
        changed[name]=any(not torch.equal(v.cpu(),old[prefix+k]) for k,v in module.named_parameters())
    smoke_ok=all(changed.values()) and all(gradients_seen[g]>0 for g in ['backbone','encoder','boxes','shared_bond','line_head','band_head']) and max_grad>0 and acceptable(metrics,baseline)
    result=dict(status=('passed' if smoke_ok else 'failed') if a.max_steps else 'completed',reason=reason,steps=step,epochs_completed=len(epoch_metrics),seconds=time.time()-start,
        peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3,gradients_seen=dict(gradients_seen),changed_modules=changed,baseline=baseline,final=metrics,best=best_metrics,
        train_images=len(train),validation_images=len(val),checkpoint_fallback_retained=True)
    write(a.output/'results.json',result);write(a.output/'progress.json',dict(status=result['status'],step=step,reason=reason,seconds=result['seconds']));print('DONE',json.dumps(result),flush=True)
    render_monitor(a.output)
    if a.max_steps and not smoke_ok:raise RuntimeError('Full-head smoke or regression guard failed')


if __name__=='__main__':main()
