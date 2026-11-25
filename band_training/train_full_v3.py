"""Train paired box + band/line outputs, with a bounded GPU smoke mode.

The full explicit training manifest is shuffled once per epoch. A max-steps
limit bounds smoke execution without replacing training with a prefix subset.
"""
import argparse,copy,csv,json,math,random,time,shutil
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from PIL import Image,ImageDraw
from .continue_indigo import ROOT,JointModel,loader,write,sha,cpu_state,match_boxes
from .train import load_model,decode_atoms,charge_exact
from engine.core import YAMLConfig
from .model import decode
from .band_line_model import BandLineDecoder,paired_bond_loss
from .full_v3_data import FullV3Dataset
from .charge_loss import ChargeAwareCriterion
from .charge_association import assign_charges
from .geometry import suppress,assemble_band_lines,edge_key,target_edges,aggregate


def atomic_checkpoint(state,path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(state,temporary)
    temporary.replace(path)


def device_targets(metas,device):
    boxes=[{k:v.to(device) for k,v in m['boxes'].items()} for m in metas]
    bands=[(m['band'][0].to(device),[(l,y,x,o.to(device),q.to(device)) for l,y,x,o,q in m['band'][1]]) for m in metas]
    return boxes,bands,[m['lines'] for m in metas]


def stratified_indices(dataset,assets,count,seed):
    """Bound smoke validation, covering charge/color/stereo/large structures."""
    if not count or count>=len(dataset):return list(range(len(dataset)))
    info={a['image']:a for a in assets};rng=random.Random(seed);allids=list(range(len(dataset)))
    rng.shuffle(allids);selected=[]
    pools=[sorted(allids,key=lambda i:-info[dataset.records[i]['image']]['atoms'])[:2],
           [i for i in allids if info[dataset.records[i]['image']]['color']][:4],
           [i for i in allids if any(str(k) in info[dataset.records[i]['image']]['bond_classes'] for k in [4,5])][:4],
           [i for i in allids if info[dataset.records[i]['image']]['charges']][:max(4,count//3)],allids]
    for pool in pools:
        for i in pool:
            if i not in selected:selected.append(i)
            if len(selected)>=count:return selected
    return selected


def metric_row(instances,ann,atoms,mapping,method,radius):
    graph=assemble_band_lines([a['bbox'] for a in atoms],instances,line_association=method,endpoint_cluster_radius=radius)
    pred=Counter();invalid=len(graph['unlinked_band_indices'])
    for b in graph['bonds']:
        a,c=mapping.get(b['begin'],-1),mapping.get(b['end'],-1)
        if min(a,c)<0:invalid+=1
        else:pred[edge_key(a,c,b['label'])]+=1
    truth=target_edges(ann);tp=sum((truth&pred).values())
    return dict(tp=tp,fp=len(instances)-tp,fn=sum(truth.values())-tp,unlinked=invalid,
                exact=int(pred==truth and invalid==0),gt=sum(truth.values())),graph


def choose_association(metrics,primary):
    candidates=[(metrics['full_graph_exact'],metrics['bond_graph']['f1'],primary),
                (metrics['forward_full_graph_exact'],metrics['forward_baseline']['f1'],'forward')]
    exact,f1,method=max(candidates,key=lambda c:c[:2])
    return method,exact+.1*f1+.01*metrics['charge_f1']


def preview(images,meta,atoms,charges,instances,path,truth=False):
    base=Image.fromarray((images.detach().cpu().permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8'))
    panels=[base,base.copy(),base.copy()];d=ImageDraw.Draw(panels[1])
    for a in atoms+charges:
        x,y,w,h=a['bbox'];d.rectangle((x,y,x+w,y+h),outline='red' if a.get('label',a.get('category_id',0))>=13 else 'blue',width=1)
    d=ImageDraw.Draw(panels[2])
    for b in instances:
        q=[tuple(p) for p in b['quad']];d.line(q+[q[0]],fill='orange',width=1)
        line=[tuple(p) for p in b['line']];d.line(line,fill='green',width=1)
        for x,y in line:d.ellipse((x-2,y-2,x+2,y+2),outline='purple',width=1)
    canvas=Image.new('RGB',(1920,690),'white');d=ImageDraw.Draw(canvas)
    for i,(panel,title) in enumerate(zip(panels,['Original','Atom / charge boxes','Bond bands + ordered lines'])):
        canvas.paste(panel,(640*i,30));d.text((640*i+8,8),title,fill='black')
    d.text((8,674),('GT ' if truth else 'Prediction ')+meta['record']['sample_id'],fill='black');canvas.save(path)


@torch.no_grad()
def evaluate(model,validation,device,args,output,tag):
    model.eval();actual=[];oracle=[];forward=[];details=[];counts=Counter();epe=[];losses=[]
    for images,metas in validation:
        images=images.to(device);boxes,out=model(images)
        _,bt,lt=device_targets(metas,device)
        loss=paired_bond_loss(out,bt,lt);losses.append({k:float(v) for k,v in loss.items()})
        for bi,meta in enumerate(metas):
            ann=meta['ann'];instances=suppress(decode(out,bi,minimum=args.threshold))
            ap=decode_atoms({k:v[bi:bi+1] for k,v in boxes.items() if torch.is_tensor(v) and v.ndim})
            atoms=[a for a in ap if a['label']<13];charges=[a for a in ap if a['label']>=13]
            mapping=match_boxes(atoms,ann['atoms']);cm=match_boxes(charges,ann['charges'],min_iou=.25)
            association=getattr(args,'line_association','exclusive_box')
            row,graph=metric_row(instances,ann,atoms,mapping,association,args.endpoint_radius);actual.append(row)
            identity={i:i for i in range(len(ann['atoms']))}
            o,_=metric_row(instances,ann,ann['atoms'],identity,association,args.endpoint_radius);oracle.append(o)
            f,forward_graph=metric_row(instances,ann,atoms,mapping,'forward',0.);forward.append(f)
            assignments=assign_charges(atoms,charges,graph['bonds'],method='safe_adaptive',lines=[b['line'] for b in instances])
            ch=charge_exact(ap,atoms,mapping,ann,assignments)
            forward_assignments=assign_charges(atoms,charges,forward_graph['bonds'],method='safe_adaptive',lines=[b['line'] for b in instances])
            forward_ch=charge_exact(ap,atoms,mapping,ann,forward_assignments)
            exact=int(row['exact'] and len(mapping)==len(atoms)==len(ann['atoms']) and ch)
            counts.update(images=1,atom_pred=len(atoms),atom_gt=len(ann['atoms']),atom_tp=len(mapping),
                charge_pred=len(charges),charge_gt=len(ann['charges']),charge_tp=len(cm),full_exact=exact,
                forward_full_exact=int(f['exact'] and len(mapping)==len(atoms)==len(ann['atoms']) and forward_ch),
                charged_images=int(bool(ann['charges'])),charged_exact=int(bool(ann['charges']) and ch))
            for (label,y,x,off,q),truth in zip(bt[bi][1],lt[bi]):
                center=(torch.tensor([x,y],device=device)+out['offset'][bi,:,y,x])*4
                pred=out['line'][bi].reshape(6,4,*out['line'].shape[-2:])[label,:,y,x].reshape(2,2)*32+center
                epe.extend(torch.linalg.vector_norm(pred-torch.tensor(truth,device=device),dim=-1).cpu().tolist())
            details.append(dict(sample_id=meta['record']['sample_id'],full_exact=exact,bond=row,atoms=len(atoms),bands=len(instances),charges=len(charges),
                endpoint_audit=graph['endpoint_assignment_audit'],charge_assignments=assignments))
            if len(details)<=3:
                preview(images[bi],meta,atoms,charges,instances,output/f'{tag}_pred_{len(details):02d}.png')
                if tag=='baseline':
                    gt=[dict(label=l,quad=q,line=line) for l,q,line in zip(meta['labels'],meta['quads'],meta['lines'])]
                    preview(images[bi],meta,ann['atoms'],ann['charges'],gt,output/f'gt_alignment_{len(details):02d}.png',True)
        if counts['images']%128<len(metas):
            print('EVAL_PROGRESS',tag,counts['images'],'/',len(validation.dataset),flush=True)
    result=dict(images=counts['images'],line_association=getattr(args,'line_association','exclusive_box'),bond_graph=aggregate(actual),oracle_boxes=aggregate(oracle),forward_baseline=aggregate(forward),
        atom_f1=2*counts['atom_tp']/max(1,counts['atom_pred']+counts['atom_gt']),
        charge_f1=2*counts['charge_tp']/max(1,counts['charge_pred']+counts['charge_gt']),
        full_graph_exact=counts['full_exact']/max(1,counts['images']),counts=dict(counts),
        forward_full_graph_exact=counts['forward_full_exact']/max(1,counts['images']),
        ordered_line_epe_at_gt_detection_anchors_px=float(np.mean(epe)) if epe else None,
        paired_losses={k:float(np.mean([l[k] for l in losses])) for k in losses[0]},
        note='Smoke subset metrics are diagnostic only; EPE uses GT detection anchors, graph metrics use decoded predictions.')
    write(output/(tag+'_metrics.json'),result);write(output/(tag+'_samples.json'),details)
    print('EVAL',tag,json.dumps(result),flush=True);return result


def tiny_overfit(model,train,assets,device,args,output):
    if not args.tiny_steps:return None
    ids=stratified_indices(train,assets,4,args.seed+1);cache=[]
    model.eval()
    from torchvision.transforms import functional as TF
    for i in ids:
        image,meta=train[i];image=image[None].to(device)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            features=model.base.encoder(model.base.backbone(TF.normalize(image,[.485,.456,.406],[.229,.224,.225])))
        if isinstance(features,tuple):features=features[0]
        cache.append((image.mean(1,keepdim=True),[f.float().detach() for f in features],meta))
    head=copy.deepcopy(model.band).train();opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4)
    gray=torch.cat([c[0] for c in cache]);features=[torch.cat([c[1][level] for c in cache]) for level in range(3)]
    _,bt,lt=device_targets([c[2] for c in cache],device);history=[]
    for step in range(args.tiny_steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=head(features,gray)
        losses=paired_bond_loss({k:v.float() for k,v in out.items()},bt,lt)
        losses['total'].backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);opt.step()
        history.append({k:float(v.detach()) for k,v in losses.items()})
        if step%20==0 or step+1==args.tiny_steps:print('TINY',step+1,json.dumps(history[-1]),flush=True)
    first=float(np.mean([h['line'] for h in history[:5]]));last=float(np.mean([h['line'] for h in history[-5:]]))
    result=dict(steps=args.tiny_steps,images=[c[2]['record']['sample_id'] for c in cache],initial_line=first,final_line=last,
        relative_line_loss_reduction=1-last/max(1e-10,first),passed=bool(last<first*.9),history=history,
        note='Separate disposable head on 4 fixed training images. Its fitted weights are not used for validation or joint training.')
    write(output/'tiny_overfit.json',result)
    if not result['passed']:raise RuntimeError('Tiny line head did not learn; inspect targets/optimization')
    del head,opt,cache,features,gray;torch.cuda.empty_cache();return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--init-joint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--tiny-steps',type=int,default=0);p.add_argument('--val-limit',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--freeze-feature-epochs',type=int,default=1);p.add_argument('--smoke-unfreeze-steps',type=int,default=4)
    p.add_argument('--freeze-box-epochs',type=int,default=1)
    p.add_argument('--threshold',type=float,default=.2);p.add_argument('--endpoint-radius',type=float,default=2.)
    p.add_argument('--seed',type=int,default=20261116);p.add_argument('--resume',type=Path)
    p.add_argument('--endpoint-order',choices=['source','canonical'],default='canonical')
    p.add_argument('--line-association',choices=['exclusive_box','exclusive_box_noisy','forward'],default='exclusive_box_noisy')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=bool(args.resume));torch.set_num_threads(2)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    device=torch.device('cuda:0');audit=json.loads(args.audit.read_text());assert audit['status']=='passed'
    train=FullV3Dataset(args.dataset,'train',augment=True,endpoint_order=args.endpoint_order)
    val=FullV3Dataset(args.dataset,'val',endpoint_order=args.endpoint_order)
    for split in ['train','val']:
        assert sha(args.dataset/(split+'.csv'))==audit['manifest_sha256'][split]
        destination=args.output/(split+'.csv')
        if args.resume:assert sha(destination)==audit['manifest_sha256'][split]
        else:shutil.copyfile(args.dataset/(split+'.csv'),destination)
    assert audit['max_box_targets']<=300,'Increase box query capacity before training this dataset'
    assets=json.loads((args.audit.parent/'assets.json').read_text())
    vi=stratified_indices(val,assets,args.val_limit,args.seed)
    training=loader(train,args.batch_size,args.workers,True);validation=loader(Subset(val,vi),args.batch_size,args.workers)
    prior=json.loads((args.init_joint.parent/'config.json').read_text());base=load_model(Path(prior['base']),device).base
    previous=torch.load(args.init_joint,map_location='cpu',weights_only=False);base.load_state_dict(previous['base'],strict=True)
    band=BandLineDecoder().to(device);band.initialize_from_band(previous['band'])
    assert all(torch.equal(v.cpu(),previous['band'][k].cpu()) for k,v in band.state_dict().items() if not k.startswith('line.'))
    base.requires_grad_(True);base.line_decoder.requires_grad_(False)
    model=JointModel(base,band).to(device)
    cfg=YAMLConfig(str(ROOT/'config/train_mp123_full.yml'));criterion=ChargeAwareCriterion(cfg.criterion.box_criterion.to(device).train())
    groups=[dict(params=base.backbone.parameters(),lr=1e-6,name='backbone'),dict(params=base.encoder.parameters(),lr=2e-6,name='encoder'),
        dict(params=base.box_decoder.parameters(),lr=2e-6,name='boxes'),
        dict(params=[v for n,v in band.named_parameters() if not n.startswith('line.')],lr=5e-5,name='band'),
        dict(params=band.line.parameters(),lr=3e-4,name='line')]
    opt=torch.optim.AdamW(groups,weight_decay=1e-4)
    total=args.max_steps or len(training)*args.epochs;warmup=min(200,max(1,total//10))
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min(1,(step+1)/warmup)*(.1+.9*.5*(1+math.cos(math.pi*min(step,total)/max(1,total)))))
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(architecture='paired_band_line_v2',base=prior['base'],checkpoint_schema='paired_band_line_joint_v2',
        init_sha256=sha(args.init_joint),train_rows=len(train),val_images=len(val),evaluated_val_images=len(vi),
        loss='charge_aware_box + 2*(exist_type + center + 1*region + 5*ordered_line)',
        outputs='18-class atom/charge boxes; each bond has 6-class confidence, ordered quad, ordered atom-anchor line',
        training_sampling='uniform shuffle of all explicit rows; existing 25% charge repeats preserved',
        atom_index_supervision=False,swap_endpoint_minimum=False,geometry='RGB 640 letterbox with shared actual rounded scale_xy',
        endpoint_order_rule='ordinary: left-to-right then top-to-bottom; wedge/dash: annotated directed line order; quad ends aligned to line' if args.endpoint_order=='canonical' else 'raw source endpoint and quad order',
        line_association=args.line_association,charge_association='safe_adaptive',
        validation_note='Full validation for formal runs; stratified diagnostic subset only when val-limit is set',
        selection_guard='Formal: atom F1 within 0.005 and charge F1 within 0.03 of initial baseline; compare graph methods on predictions',
        model_limitations=['13 atom classes: unknown elements/labels map to other; text/isotope/explicit H decoding not trained',
        'GT charge association ceiling is 14973/14980, not 100%; predicted endpoints need tolerance validated on held-out data'])
    write(args.output/'config.json',config);write(args.output/'validation_sample_ids.json',[val.records[i]['sample_id'] for i in vi])
    source=args.output/'source';source.mkdir(exist_ok=True)
    for name in ['train_full_v3.py','band_line_model.py','full_v3_data.py','model.py','mixed_data.py','geometry.py','endpoint_assignment.py','charge_loss.py','charge_association.py','charge_adaptive.py']:
        shutil.copyfile(Path(__file__).with_name(name),source/name)
    start=step=0;best=-1;history=[]
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert saved['schema']=='paired_band_line_joint_v2'
        for key in ['endpoint_order','epochs','max_steps','batch_size','seed']:
            assert saved['config'][key]==config[key],f'Resume changed {key}'
        model.base.load_state_dict(saved['base']);model.band.load_state_dict(saved['band'])
        opt.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['scheduler'])
        start=saved['next_epoch'];step=saved['step'];best=saved['best_score']
        if 'rng' in saved:
            rng=saved['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda'])
    started=time.time();tiny=None;baseline=None
    if args.resume and (args.output/'baseline_metrics.json').exists():
        baseline=json.loads((args.output/'baseline_metrics.json').read_text())
    if not args.resume:
        write(args.output/'progress.json',dict(status='baseline_validation',step=0,validation_images=len(vi),total_epochs=args.epochs))
        baseline=evaluate(model,validation,device,args,args.output,'baseline')
        tiny=tiny_overfit(model,train,assets,device,args,args.output)
    log=(args.output/'history.jsonl').open('a');group_gradients={};updated_groups=set();last_metrics=None
    for epoch in range(start,args.epochs):
        for images,metas in training:
            features_trainable=(step>=max(0,args.max_steps-args.smoke_unfreeze_steps)) if args.max_steps else epoch>=args.freeze_feature_epochs
            boxes_trainable=(step>=max(0,args.max_steps-args.smoke_unfreeze_steps)) if args.max_steps else epoch>=args.freeze_box_epochs
            base.backbone.requires_grad_(features_trainable);base.encoder.requires_grad_(features_trainable)
            base.box_decoder.requires_grad_(boxes_trainable)
            model.train();base.backbone.eval();base.encoder.eval();base.line_decoder.eval()
            # The existing criterion requires training auxiliary outputs even
            # while parameters are frozen. Freeze BN running statistics too.
            if not boxes_trainable:
                for module in base.box_decoder.modules():
                    if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
            images=images.to(device);box_targets,bt,lt=device_targets(metas,device)
            opt.zero_grad(set_to_none=True);boxes,out=model(images,box_targets)
            box_losses=criterion(boxes,box_targets);bond_losses=paired_bond_loss(out,bt,lt)
            box_total=sum(box_losses.values());loss=box_total+2*bond_losses['total']
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite total loss')
            loss.backward()
            norms={g['name']:float(torch.sqrt(sum((p.grad.detach().float().square().sum() for p in g['params'] if p.grad is not None),torch.tensor(0.,device=device)))) for g in opt.param_groups}
            for name,norm in norms.items():
                if not math.isfinite(norm):raise RuntimeError('Nonfinite gradient '+name)
                if norm>0:updated_groups.add(name)
                group_gradients[name]=max(group_gradients.get(name,0),norm)
            if any(norms[n]<=0 for n in ['band','line']+(['boxes'] if boxes_trainable else [])):raise RuntimeError('Disconnected detection head')
            if features_trainable and any(norms[n]<=0 for n in ['backbone','encoder']):raise RuntimeError('Disconnected features')
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();schedule.step();step+=1
            row=dict(step=step,epoch=epoch+1,feature_gradients=features_trainable,box_gradients=boxes_trainable,total=float(loss.detach()),box=float(box_total.detach()),
                **{k:float(v.detach()) for k,v in bond_losses.items() if k!='total'},gradient_norms=norms,
                charge_images=sum(bool(m['ann']['charges']) for m in metas),seconds=time.time()-started)
            log.write(json.dumps(row)+'\n');log.flush();history.append(row)
            if step==1 or step%10==0 or (args.max_steps and step==args.max_steps):
                print('TRAIN',json.dumps(row),flush=True)
                write(args.output/'progress.json',dict(status='training',step=step,epoch=epoch+1,total_epochs=args.epochs,
                    total_steps=total,steps_per_epoch=len(training),**{k:v for k,v in row.items() if k not in ['step','epoch']}))
            if args.max_steps and step>=args.max_steps:break
        last_metrics=evaluate(model,validation,device,args,args.output,f'epoch_{epoch+1:02d}')
        selected_association,score=choose_association(last_metrics,args.line_association)
        if not args.max_steps and baseline and (last_metrics['atom_f1']<baseline['atom_f1']-.005 or last_metrics['charge_f1']<baseline['charge_f1']-.03):
            score=-1.
        isbest=score>best;best=max(best,score)
        inference_config=dict(config,line_association=selected_association,validation_primary_association=args.line_association)
        state=dict(schema='paired_band_line_joint_v2',base=cpu_state(base),band=cpu_state(band),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),
            step=step,next_epoch=epoch+1,best_score=best,metrics=last_metrics,config=inference_config,threshold=args.threshold,
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
        atomic_checkpoint(state,args.output/'last_joint.pt')
        if isbest:atomic_checkpoint(state,args.output/'best_joint.pt')
        write(args.output/'progress.json',dict(status='training',step=step,epoch=epoch+1,metrics=last_metrics))
        if args.max_steps and step>=args.max_steps:break
    log.close()
    # Round-trip every learned tensor, including the newly added line head.
    saved=torch.load(args.output/'last_joint.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v.cpu(),saved['band'][k]) for k,v in band.state_dict().items())
    assert all(torch.equal(v.cpu(),saved['base'][k]) for k,v in base.state_dict().items())
    result=dict(status='passed' if args.max_steps else 'completed',mode='smoke' if args.max_steps else 'training',steps=step,
        training_sampling_rows=len(train),unique_training_images=audit['unique_train'],validation_images=len(val),evaluated_val_images=len(vi),
        seconds=time.time()-started,peak_gpu_memory_gb=torch.cuda.max_memory_allocated()/1024**3,
        selected_line_association=saved['config']['line_association'],
        tiny_overfit={k:v for k,v in tiny.items() if k!='history'} if tiny else None,
        baseline=baseline,final=last_metrics,updated_groups=sorted(updated_groups),max_gradient_norms=group_gradients,
        checkpoint_roundtrip='all base and paired bond tensors exactly equal',
        note='Smoke verifies plumbing, gradients, tiny fitting and prediction assembly; it does not establish trained-model accuracy.')
    write(args.output/'results.json',result);write(args.output/'progress.json',dict(status=result['status'],step=step));print('DONE',json.dumps(result),flush=True)

if __name__=='__main__':main()
