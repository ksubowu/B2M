"""Warm-start independent scored line/band heads; bounded smoke or full training."""
import argparse,copy,json,math,random,time,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torchvision.transforms import functional as TF
from PIL import Image,ImageDraw
from scipy.optimize import linear_sum_assignment
from .train import load_model,decode_atoms,charge_exact,ROOT
from .continue_indigo import loader,write,sha,cpu_state,match_boxes
from .train_full_v3 import stratified_indices,atomic_checkpoint
from .scored_bond_data import ScoredBondDataset
from .scored_bond_model import ScoredBondDecoder,move_scored_targets,scored_bond_loss,decode_scored_bonds
from .scored_bond_postprocess import assemble_scored_bonds,group_lines,refine_peak_lines
from .geometry import edge_key,target_edges,aggregate,polygon_iou
from .charge_association import assign_charges
from .charge_loss import ChargeAwareCriterion
from engine.core import YAMLConfig

SCHEMA='independent_scored_line_band_joint_v3'


class ScoredJointModel(nn.Module):
    def __init__(self,base,decoder):super().__init__();self.base=base;self.decoder=decoder

    def features(self,images):
        with torch.autocast(images.device.type,dtype=torch.bfloat16,enabled=images.is_cuda):
            f=self.base.encoder(self.base.backbone(TF.normalize(images,[.485,.456,.406],[.229,.224,.225])))
        if isinstance(f,tuple):f=f[0]
        return [v.float() for v in f]

    def forward(self,images,targets=None):
        f=self.features(images);boxes=self.base.box_decoder(f,targets=targets)
        with torch.autocast(images.device.type,dtype=torch.bfloat16,enabled=images.is_cuda):out=self.decoder(f,images.mean(1,keepdim=True))
        return boxes,{name:{k:v.float() for k,v in head.items()} for name,head in out.items()}


def preview(image,meta,atoms,charges,decoded,path):
    base=Image.fromarray((image.detach().cpu().permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8'))
    panels=[base,base.copy(),base.copy(),base.copy()]
    draw=ImageDraw.Draw(panels[1])
    for box in atoms+charges:
        x,y,w,h=box['bbox'];draw.rectangle((x,y,x+w,y+h),outline='red' if box['label']>=13 else 'blue')
    draw=ImageDraw.Draw(panels[2])
    for item in decoded['lines']:
        points=[tuple(v) for v in item['line']];draw.line(points,fill='green',width=1)
        for x,y in points:draw.ellipse((x-2,y-2,x+2,y+2),outline='purple')
    draw=ImageDraw.Draw(panels[3])
    for item in decoded['bands']:
        points=[tuple(v) for v in item['quad']];draw.line(points+[points[0]],fill='orange',width=1)
    canvas=Image.new('RGB',(2560,690),'white');draw=ImageDraw.Draw(canvas)
    for j,(panel,title) in enumerate(zip(panels,['Original','Atom / charge boxes','Independent lines + line score','Independent bands + band score'])):
        canvas.paste(panel,(640*j,30));draw.text((640*j+8,8),title,fill='black')
    draw.text((8,676),meta['record']['sample_id'],fill='black');canvas.save(path)


def typed_metric(graph,ann,mapping):
    truth=target_edges(ann);pred=Counter();invalid=len(graph['unlinked_band_indices'])+len(graph['untyped_lines'])
    count=len(graph['bonds'])+invalid
    for e in graph['bonds']:
        a,b=mapping.get(e['begin'],-1),mapping.get(e['end'],-1)
        if min(a,b)<0:invalid+=1
        else:pred[edge_key(a,b,e['label'])]+=1
    tp=sum((truth&pred).values())
    return dict(tp=tp,fp=count-tp,fn=sum(truth.values())-tp,unlinked=invalid,exact=int(pred==truth and not invalid),gt=sum(truth.values()))


@torch.no_grad()
def evaluate(model,validation,device,args,out,tag):
    model.eval();counts=Counter();typed=[];loss_values=[];details=[]
    for images,metas in validation:
        images=images.to(device);boxes,output=model(images)
        losses=scored_bond_loss(output,move_scored_targets([m['scored_bonds'] for m in metas],device));loss_values.append({k:float(v) for k,v in losses.items()})
        for bi,meta in enumerate(metas):
            refine=getattr(args,'refine_line_peaks',False)
            decoded=decode_scored_bonds(output,bi,args.line_threshold,args.band_threshold,apply_line_peaks=not refine)
            # Measure the head's peak geometry before box-assisted refinement.
            # Otherwise a postprocessing gain would be mislabeled a head gain.
            raw_peak_lines=[r for r in decoded['lines'] if r.get('is_local_peak',True)]
            segments=[g[0]['line'] for g in group_lines(raw_peak_lines)]
            ap=decode_atoms({k:v[bi:bi+1] for k,v in boxes.items() if torch.is_tensor(v) and v.ndim})
            atoms=[a for a in ap if a['label']<13];charges=[a for a in ap if a['label']>=13]
            mapping=match_boxes(atoms,meta['ann']['atoms']);cm=match_boxes(charges,meta['ann']['charges'],min_iou=.25)
            if refine:decoded=refine_peak_lines([a['bbox'] for a in atoms],decoded,args.line_threshold)
            graph=assemble_scored_bonds([a['bbox'] for a in atoms],decoded,stereo_from_band=getattr(args,'stereo_from_band',False))
            # Head-specific geometry metrics do not use atom detections.
            line_tp=0
            if segments and meta['lines']:
                distances=np.linalg.norm(np.asarray(segments)[:,None]-np.asarray(meta['lines'])[None],axis=-1)
                cost=distances.mean(-1);ii,jj=linear_sum_assignment(cost)
                line_tp=sum(cost[i,j]<=5 and distances[i,j].max()<=8 for i,j in zip(ii,jj))
            band_tp=0
            if decoded['bands'] and meta['quads']:
                iou=np.array([[polygon_iou(p['quad'],q) for q in meta['quads']] for p in decoded['bands']])
                same=np.array([p['label'] for p in decoded['bands']])[:,None]==np.array(meta['labels'])[None]
                ii,jj=linear_sum_assignment(1-iou+1000*(~same));band_tp=sum(same[i,j] and iou[i,j]>=.25 for i,j in zip(ii,jj))
            metric=typed_metric(graph,meta['ann'],mapping);typed.append(metric)
            expected=Counter(tuple(sorted((b['begin'],b['end']))) for b in meta['ann']['bonds']);actual=Counter();line_invalid=len(graph['unlinked_lines'])
            for edge in graph['line_graph']:
                a,b=mapping.get(edge['begin'],-1),mapping.get(edge['end'],-1)
                if min(a,b)<0:line_invalid+=1
                else:actual[tuple(sorted((a,b)))]+=1
            hit=sum((expected&actual).values())
            assignments=assign_charges(atoms,charges,graph['bonds'],method='safe_adaptive',lines=[r['line'] for r in decoded['lines']])
            charge=charge_exact(ap,atoms,mapping,meta['ann'],assignments)
            atom_exact=len(mapping)==len(atoms)==len(meta['ann']['atoms'])
            counts.update(images=1,line_pred=len(graph['line_graph'])+len(graph['unlinked_lines']),line_gt=sum(expected.values()),line_tp=hit,
                geometric_line_tp=int(line_tp),geometric_line_pred=len(segments),geometric_line_gt=len(meta['lines']),
                band_region_tp=int(band_tp),band_region_pred=len(decoded['bands']),band_region_gt=len(meta['quads']),
                line_graph_exact=int(actual==expected and not line_invalid),untyped_lines=len(graph['untyped_lines']),band_only=len(graph['band_only_candidates']),
                atom_pred=len(atoms),atom_gt=len(meta['ann']['atoms']),atom_tp=len(mapping),charge_pred=len(charges),charge_gt=len(meta['ann']['charges']),charge_tp=len(cm),
                full_graph_exact=int(atom_exact and charge and metric['exact']))
            details.append(dict(sample_id=meta['record']['sample_id'],lines=len(decoded['lines']),bands=len(decoded['bands']),typed_edges=len(graph['bonds']),
                untyped_lines=len(graph['untyped_lines']),band_only=len(graph['band_only_candidates']),typed_metric=metric))
            if len(details)<=2:preview(images[bi],meta,atoms,charges,decoded,out/f'{tag}_{len(details):02d}.png')
        if counts['images']%128<len(metas):print('EVAL_PROGRESS',tag,counts['images'],flush=True)
    result=dict(images=counts['images'],counts=dict(counts),typed_bond=aggregate(typed),
        line_geometry_f1=2*counts['geometric_line_tp']/max(1,counts['geometric_line_pred']+counts['geometric_line_gt']),
        line_geometry_match='fixed-order endpoint mean <=5px and max <=8px, one-to-one; independent of atom detections',
        band_region_type_f1_iou025=2*counts['band_region_tp']/max(1,counts['band_region_pred']+counts['band_region_gt']),
        untyped_line_detection_f1=2*counts['line_tp']/max(1,counts['line_pred']+counts['line_gt']),
        atom_f1=2*counts['atom_tp']/max(1,counts['atom_pred']+counts['atom_gt']),charge_f1=2*counts['charge_tp']/max(1,counts['charge_pred']+counts['charge_gt']),
        full_graph_exact=counts['full_graph_exact']/max(1,counts['images']),losses={k:float(np.mean([r[k] for r in loss_values])) for k in loss_values[0]},
        score_definition='line_score=sigmoid(line objectness); band_score=sigmoid(band objectness)*softmax(type)[selected class]',
        postprocess=dict(refine_line_peaks=getattr(args,'refine_line_peaks',False),stereo_from_band=getattr(args,'stereo_from_band',False)),
        scope='Smoke subset diagnostic when val-limit>0; new independent scores require retraining/calibration, not comparable as final model accuracy.')
    write(out/(tag+'_metrics.json'),result);write(out/(tag+'_samples.json'),details);print('EVAL',tag,json.dumps(result),flush=True);return result


def tiny_fit(model,train,assets,device,args):
    if not args.tiny_steps:return None
    ids=stratified_indices(train,assets,4,args.seed);model.eval();cache=[]
    for i in ids:
        image,meta=train[i];image=image[None].to(device)
        with torch.no_grad():f=model.features(image)
        cache.append((f,image.mean(1,keepdim=True),meta))
    f=[torch.cat([r[0][level] for r in cache]) for level in range(3)];gray=torch.cat([r[1] for r in cache])
    targets=move_scored_targets([r[2]['scored_bonds'] for r in cache],device)
    head=copy.deepcopy(model.decoder).train();opt=torch.optim.AdamW(head.parameters(),lr=3e-4);history=[]
    for step in range(args.tiny_steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):output=head(f,gray)
        output={name:{k:v.float() for k,v in h.items()} for name,h in output.items()};loss=scored_bond_loss(output,targets)
        loss['total'].backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);opt.step()
        history.append({k:float(v.detach()) for k,v in loss.items()})
        if step%25==0 or step+1==args.tiny_steps:print('TINY',step+1,json.dumps(history[-1]),flush=True)
    first={k:float(np.mean([r[k] for r in history[:5]])) for k in history[0]};last={k:float(np.mean([r[k] for r in history[-5:]])) for k in history[0]}
    passed=last['line_score']<first['line_score']*.9 and last['band_score']<first['band_score']*.9 and last['total']<first['total']*.8
    result=dict(passed=passed,initial=first,final=last,steps=args.tiny_steps,training_sample_ids=[r[2]['record']['sample_id'] for r in cache],history=history,
        scope='Separate temporary decoder overfits 4 training images; its weights are not used for main training/validation.')
    write(args.output/'tiny_overfit.json',result)
    if not passed:raise RuntimeError('Independent score heads did not learn on the tiny training set')
    del head,opt,cache,f,gray;torch.cuda.empty_cache();return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--init',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--target-audit',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--max-steps',type=int,default=0);p.add_argument('--tiny-steps',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--workers',type=int,default=4);p.add_argument('--val-limit',type=int,default=0)
    p.add_argument('--freeze-base-epochs',type=int,default=1);p.add_argument('--line-threshold',type=float,default=.2);p.add_argument('--band-threshold',type=float,default=.2)
    p.add_argument('--seed',type=int,default=20261117);p.add_argument('--resume',type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=bool(args.resume));audit=json.loads(args.target_audit.read_text());assert audit['status']=='passed'
    torch.set_num_threads(2);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;device=torch.device('cuda:0')
    source=torch.load(args.init,map_location='cpu',weights_only=False);assert source['schema']=='paired_band_line_joint_v2'
    train=ScoredBondDataset(args.dataset,'train',augment=True);val=ScoredBondDataset(args.dataset,'val')
    hashes={name:sha(args.dataset/(name+'.csv')) for name in ['train','val']}
    for name in ['train','val']:
        dst=args.output/(name+'.csv')
        if args.resume:assert sha(dst)==hashes[name]
        else:shutil.copyfile(args.dataset/(name+'.csv'),dst)
    assets=json.loads((ROOT/'runs/full_v3_training_audit_20261116/assets.json').read_text())
    vi=stratified_indices(val,assets,args.val_limit,args.seed);training=loader(train,args.batch_size,args.workers,True);validation=loader(Subset(val,vi),args.batch_size,args.workers)
    base=load_model(Path(source['config']['base']),device).base;base.load_state_dict(source['base'],strict=True)
    decoder=ScoredBondDecoder().to(device);decoder.initialize_from_paired_v2(source['band']);model=ScoredJointModel(base,decoder).to(device)
    shared=list(decoder.project.parameters())+list(decoder.image.parameters())+list(decoder.body.parameters())
    groups=[dict(params=shared,lr=2e-5,name='shared_bond'),dict(params=decoder.line_head.parameters(),lr=3e-4,name='line_head'),dict(params=decoder.band_head.parameters(),lr=1e-4,name='band_head'),
        dict(params=base.backbone.parameters(),lr=1e-6,name='backbone'),dict(params=base.encoder.parameters(),lr=2e-6,name='encoder'),dict(params=base.box_decoder.parameters(),lr=2e-6,name='boxes')]
    opt=torch.optim.AdamW(groups,weight_decay=1e-4);total=args.max_steps or len(training)*args.epochs;warmup=min(200,max(1,total//10))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda i:min(1,(i+1)/warmup)*(.1+.9*.5*(1+math.cos(math.pi*min(i,total)/total))))
    cfg=YAMLConfig(str(ROOT/'config/train_mp123_full.yml'));criterion=ChargeAwareCriterion(cfg.criterion.box_criterion.to(device).train())
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()};config.update(schema=SCHEMA,base=source['config']['base'],
        train_rows=len(train),unique_train=len({r.get('asset_sample_id') or r['sample_id'].split('::repeat_')[0] for r in train.records}),validation_images=len(val),evaluated_val_images=len(vi),manifest_sha256=hashes,init_sha256=sha(args.init),
        pairing='Shared spatial instance query, independent heads/scores/geometry. Line selection never uses band score.',
        losses='line focal + line offset + 5*ordered endpoints; band focal + categorical type CE + band offset + quad regression',
        no_atom_index_targets=True,score_definitions=dict(line='sigmoid(line objectness)',band='sigmoid(band objectness)*softmax(type)[selected class]'))
    write(args.output/'config.json',config);write(args.output/'validation_sample_ids.json',[val.records[i]['sample_id'] for i in vi])
    snapshots=args.output/'source';snapshots.mkdir(exist_ok=True)
    for name in ['scored_bond_model.py','scored_bond_data.py','scored_bond_postprocess.py','train_scored_bonds.py']:
        shutil.copyfile(Path(__file__).with_name(name),snapshots/name)
    epoch_start=step=0;best=-1.;baseline=None
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False);assert saved['schema']==SCHEMA
        for key in ['manifest_sha256','epochs','batch_size','max_steps']:assert saved['config'][key]==config[key]
        base.load_state_dict(saved['base']);decoder.load_state_dict(saved['decoder']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        epoch_start=saved['next_epoch'];step=saved['step'];best=saved['best_score']
        baseline=json.loads((args.output/'baseline_metrics.json').read_text())
        if 'rng' in saved:
            rng=saved['rng'];torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
    started=time.time()
    if not args.resume:
        write(args.output/'progress.json',dict(status='baseline_validation',step=0,total_epochs=args.epochs,validation_images=len(vi)))
        baseline=evaluate(model,validation,device,args,args.output,'baseline')
    tiny=tiny_fit(model,train,assets,device,args) if not args.resume else None
    max_grad=Counter();history=[]
    with (args.output/'history.jsonl').open('a') as log:
        for epoch in range(epoch_start,args.epochs):
            train_base=epoch>=args.freeze_base_epochs;base.requires_grad_(train_base);base.line_decoder.requires_grad_(False)
            for images,metas in training:
                model.train();base.backbone.eval();base.encoder.eval();base.line_decoder.eval()
                if not train_base:base.eval()
                images=images.to(device);targets=move_scored_targets([m['scored_bonds'] for m in metas],device)
                bt=[{k:v.to(device) for k,v in m['boxes'].items()} for m in metas]
                opt.zero_grad(set_to_none=True);boxes,output=model(images,bt if train_base else None)
                losses=scored_bond_loss(output,targets);box_loss=sum(criterion(boxes,bt).values()) if train_base else losses['total']*0
                total_loss=2*losses['total']+box_loss
                if not torch.isfinite(total_loss):raise RuntimeError('Nonfinite independent-head loss')
                total_loss.backward();norms={g['name']:float(torch.sqrt(sum((p.grad.float().square().sum() for p in g['params'] if p.grad is not None),torch.tensor(0.,device=device)))) for g in opt.param_groups}
                for name in ['line_head','band_head','shared_bond']:
                    if not math.isfinite(norms[name]) or norms[name]<=0:raise RuntimeError('Invalid gradient '+name)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();scheduler.step();step+=1
                for name,value in norms.items():max_grad[name]=max(max_grad[name],value)
                row=dict(step=step,epoch=epoch+1,base_trainable=train_base,gradient_norms=norms,**{k:float(v.detach()) for k,v in losses.items()})
                history.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if step==1 or step%10==0 or step==args.max_steps:
                    print('TRAIN',json.dumps(row),flush=True)
                    write(args.output/'progress.json',dict(status='training',step=step,epoch=epoch+1,total_epochs=args.epochs,total_steps=total,
                        steps_per_epoch=len(training),seconds=time.time()-started,base_trainable=train_base,
                        line_loss=row['line_total'],band_loss=row['band_total'],gradient_norms=norms))
                if args.max_steps and step>=args.max_steps:break
            write(args.output/'progress.json',dict(status='epoch_validation',step=step,epoch=epoch+1,total_epochs=args.epochs,validation_images=len(vi)))
            metrics=evaluate(model,validation,device,args,args.output,f'epoch_{epoch+1:02d}')
            score=metrics['full_graph_exact']+.1*metrics['typed_bond']['f1']+.01*metrics['untyped_line_detection_f1']
            if not args.max_steps and baseline and (metrics['atom_f1']<baseline['atom_f1']-.005 or metrics['charge_f1']<baseline['charge_f1']-.03):score=-1.
            improved=score>best;best=max(best,score)
            state=dict(schema=SCHEMA,base=cpu_state(base),decoder=cpu_state(decoder),step=step,next_epoch=epoch+1,best_score=best,config=config,metrics=metrics,
                optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()))
            atomic_checkpoint(state,args.output/'last_joint.pt')
            if improved:atomic_checkpoint(state,args.output/'best_joint.pt')
            if args.max_steps and step>=args.max_steps:break
    saved=torch.load(args.output/'last_joint.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v.cpu(),saved['decoder'][k]) for k,v in decoder.state_dict().items())
    if not train_base:assert all(torch.equal(v.cpu(),source['base'][k]) for k,v in base.state_dict().items()),'Frozen atom/charge/base weights changed'
    result=dict(status='passed' if args.max_steps else 'completed',mode='smoke' if args.max_steps else 'training',steps=step,seconds=time.time()-started,
        peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3,baseline=baseline,final=metrics,tiny_overfit={k:v for k,v in tiny.items() if k!='history'} if tiny else None,
        max_gradient_norms=dict(max_grad),checkpoint_roundtrip=True,base_weights_unchanged=not train_base,
        scope='Architecture/optimization smoke only; new confidence heads require full training/calibration. Existing v2 production weights untouched.')
    write(args.output/'results.json',result);write(args.output/'progress.json',dict(status=result['status'],step=step));print('DONE',json.dumps(result),flush=True)

if __name__=='__main__':main()
