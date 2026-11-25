"""Continue independent atom/charge and band heads on the existing Indigo split.

All image/box/quad coordinates use one RGB letterbox transform. Band loss sees
only labels and ordered region geometry. Atom identities are evaluation-only.
"""
import argparse,hashlib,json,math,random,time,shutil
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from PIL import Image,ImageDraw,ImageFont
from .train import ROOT,load_model,decode_atoms,charge_exact
from engine.core import YAMLConfig
from .model import BandDecoder,band_loss,decode
from .mixed_data import MixedIndigoDataset,collate,worker_init
from .geometry import edge_metrics,aggregate,suppress,pred_edges,target_edges,link_many
from .charge_loss import ChargeAwareCriterion
from .charge_association import assign_charges


def write(path,data):
    def encode(value):
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,Path):return str(value)
        raise TypeError(type(value).__name__+' is not JSON serializable')
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=encode));tmp.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def iou_matrix(boxes,truth):
    a=np.asarray(boxes,float).reshape(-1,4);b=np.asarray(truth,float).reshape(-1,4)
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    al=a[:,:2];ah=al+a[:,2:];bl=b[:,:2];bh=bl+b[:,2:]
    intersection=np.maximum(0,np.minimum(ah[:,None],bh[None])-np.maximum(al[:,None],bl[None])).prod(2)
    return intersection/np.maximum(1e-8,(a[:,2:]).prod(1)[:,None]+(b[:,2:]).prod(1)[None]-intersection)


def match_boxes(predicted,truth,labels='category_id',min_iou=.1):
    from scipy.optimize import linear_sum_assignment
    if not predicted or not truth:return {}
    boxes=[p['bbox'] for p in predicted];gt=[g['bbox'] for g in truth]
    iou=iou_matrix(boxes,gt)
    centers=np.array([np.array(b[:2])+np.array(b[2:])/2 for b in boxes])
    gc=np.array([np.array(b[:2])+np.array(b[2:])/2 for b in gt])
    dist=np.linalg.norm(centers[:,None]-gc[None],axis=2)
    diagonal=np.linalg.norm(np.asarray(gt)[:,2:],axis=1)
    label_match=np.array([p['label'] for p in predicted])[:,None]==np.array([g[labels] for g in truth])[None]
    cost=3*(1-iou)+dist/(4+diagonal[None]/2)+1000*(~label_match)
    rows,cols=linear_sum_assignment(cost)
    return {int(i):int(j) for i,j in zip(rows,cols) if label_match[i,j] and
            (iou[i,j]>=min_iou or (min_iou<.2 and dist[i,j]<max(8,diagonal[j]*.5)))}


class JointModel(nn.Module):
    def __init__(self,base,band):super().__init__();self.base=base;self.band=band
    def forward(self,images,targets=None):
        normalized=TF.normalize(images,[.485,.456,.406],[.229,.224,.225])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f=self.base.encoder(self.base.backbone(normalized))
        if isinstance(f,tuple):f=f[0]
        f=[v.float() for v in f]
        boxes=self.base.box_decoder(f,targets=targets)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            bands=self.band(f,images.mean(1,keepdim=True))
        return boxes,{k:v.float() for k,v in bands.items()}


def loader(dataset,batch,workers,shuffle=False):
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=workers,
                      pin_memory=True,collate_fn=collate,worker_init_fn=worker_init,
                      persistent_workers=workers>0)


def metric_summary(rows,atom_counts,charge_counts,strict,charge_exact_count,charged_exact,charged_images):
    ap,at,ag=map(int,atom_counts);cp,ct,cg=map(int,charge_counts)
    return dict(bond_graph=aggregate(rows),atom_precision=at/max(1,ap),atom_recall=at/max(1,ag),
                atom_f1=2*at/max(1,ap+ag),charge_precision=ct/max(1,cp),charge_recall=ct/max(1,cg),
                charge_f1=2*ct/max(1,cp+cg),charge_gt=cg,charge_pred=cp,charge_tp=ct,
                full_graph_exact=strict/max(1,len(rows)),charge_graph_exact=charge_exact_count/max(1,len(rows)),
                charged_image_exact=charged_exact/max(1,charged_images),charged_images=charged_images)


@torch.no_grad()
def evaluate(model,dataloader,device,threshold,output,tag,preview_count=6,association='kdtree',charge_association='legacy'):
    model.eval();oracle=[];actual=[];ceil=[];atom_counts=np.zeros(3,int);charge_counts=np.zeros(3,int)
    strict=charge_ok=charged_exact=charged_images=0;charge_classes=defaultdict(Counter);previewed=charged_previews=0
    charge_localized=0
    buckets=defaultdict(list);types=defaultdict(Counter);failures=[];details=[];started=time.time();done=0
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for images,metas in dataloader:
        images=images.to(device,non_blocking=True);boxout,bandout=model(images)
        for bi,meta in enumerate(metas):
            ann=meta['ann'];raw=decode(bandout,bi,minimum=threshold);ps=suppress(raw)
            ap=decode_atoms({k:v[bi:bi+1] for k,v in boxout.items() if isinstance(v,torch.Tensor) and v.ndim>=1})
            atoms=[a for a in ap if a['label']<13];charges=[q for q in ap if q['label']>=13]
            mapping=match_boxes(atoms,ann['atoms']);cm=match_boxes(charges,ann['charges'],min_iou=.25)
            for q in ann['charges']:charge_classes[q['category_id']]['gt']+=1
            for q in charges:charge_classes[q['label']]['pred']+=1
            for pi in cm:charge_classes[charges[pi]['label']]['tp']+=1
            if ann['charges']:
                query_boxes=boxout['pred_boxes'][bi].detach().float().cpu().numpy()*640
                query_boxes[:,:2]-=query_boxes[:,2:]/2
                geometry_predictions=[dict(label=0,bbox=b.tolist()) for b in query_boxes]
                geometry_truth=[dict(category_id=0,bbox=q['bbox']) for q in ann['charges']]
                charge_localized+=len(match_boxes(geometry_predictions,geometry_truth,min_iou=.25))
            atom_counts+=np.array([len(atoms),len(mapping),len(ann['atoms'])]);charge_counts+=np.array([len(charges),len(cm),len(ann['charges'])])
            row=edge_metrics(ps,ann,[a['bbox'] for a in atoms],mapping,method=association);actual.append(row)
            o=edge_metrics(ps,ann,[a['bbox'] for a in ann['atoms']],method=association);oracle.append(o)
            gtps=[dict(label=l,quad=q,score=1.) for l,q in zip(meta['labels'],meta['quads'])]
            ceil.append(edge_metrics(gtps,ann,[a['bbox'] for a in ann['atoms']],method=association))
            charge_links=None
            if charge_association!='legacy':
                pairs=link_many([p['quad'] for p in ps],[a['bbox'] for a in atoms],method=association)
                geometric_bonds=[dict(begin=pair[0],end=pair[1],label=p['label']) for p,pair in zip(ps,pairs) if pair is not None]
                charge_links=assign_charges(atoms,charges,geometric_bonds,method=charge_association)
            ch=charge_exact(ap,atoms,mapping,ann,assignments=charge_links);charge_ok+=ch
            charged_images+=bool(ann['charges']);charged_exact+=int(ch and bool(ann['charges']))
            exact=int(row['exact'] and len(mapping)==len(atoms)==len(ann['atoms']) and ch);strict+=exact
            n=len(ann['atoms']);bucket='<=25' if n<=25 else '26-50' if n<=50 else '51-100' if n<=100 else '>100'
            buckets[bucket].append(row)
            truth=target_edges(ann);pred,_=pred_edges(ps,[a['bbox'] for a in ann['atoms']],method=association)
            hit=truth&pred
            for key,count in truth.items():types[key[2]]['gt']+=count
            for key,count in pred.items():types[key[2]]['pred']+=count
            for key,count in hit.items():types[key[2]]['tp']+=count
            detail=dict(sample_id=meta['record']['sample_id'],dataset=meta['record']['dataset'],
                        color=meta['record']['is_color'],atoms_gt=n,bonds_gt=len(ann['bonds']),
                        atoms_pred=len(atoms),bands_pred=len(ps),candidates_before_nms=len(raw),
                        bond_f1=2*row['tp']/max(1,2*row['tp']+row['fp']+row['fn']),full_exact=exact,
                        oracle_bond_f1=2*o['tp']/max(1,2*o['tp']+o['fp']+o['fn']))
            detail.update(charges_gt=len(ann['charges']),charges_pred=len(charges),charges_tp=len(cm))
            detail.update(charge_association_exact=bool(ch),charge_assignments=charge_links,
                          legacy_charge_exact=bool(charge_exact(ap,atoms,mapping,ann)))
            details.append(detail)
            if not row['exact']:failures.append(detail)
            if previewed<preview_count and (done<2 or (ann['charges'] and charged_previews<preview_count-2)):
                rgb=(images[bi].cpu().permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')
                base=Image.fromarray(rgb);truth_image=base.copy();pred_image=base.copy();td=ImageDraw.Draw(truth_image);pd=ImageDraw.Draw(pred_image)
                for q in meta['quads']:
                    poly=[tuple(p) for p in q];td.line(poly+[poly[0]],fill='green',width=1)
                for a in ann['atoms']+ann['charges']:
                    x,y,w,h=a['bbox'];td.rectangle((x,y,x+w,y+h),outline='blue' if a['category_id']<13 else 'red',width=1)
                for p in ps:
                    poly=[tuple(v) for v in p['quad']];pd.line(poly+[poly[0]],fill='orange',width=1)
                for a in ap:
                    x,y,w,h=a['bbox'];pd.rectangle((x,y,x+w,y+h),outline='blue' if a['label']<13 else 'red',width=1)
                canvas=Image.new('RGB',(1920,720),'white');d=ImageDraw.Draw(canvas)
                for j,(im,title) in enumerate(zip([base,truth_image,pred_image],['Validation RGB image','Ground truth: boxes + bands','Predicted: boxes + bands'])):
                    canvas.paste(im,(j*640,40));d.text((j*640+8,8),title,font=font,fill='black')
                d.text((8,686),meta['record']['sample_id'],font=font,fill='black');canvas.save(output/f'{tag}_sample_{previewed:02d}.png')
                previewed+=1;charged_previews+=bool(ann['charges'])
            done+=1
        if done%128<len(metas):print(f'EVAL {tag}: {done}/{len(dataloader.dataset)}',flush=True)
    summary=metric_summary(actual,atom_counts,charge_counts,strict,charge_ok,charged_exact,charged_images)
    summary.update(oracle_atoms=aggregate(oracle),geometry_ceiling=aggregate(ceil),
                   complexity={k:aggregate(v) for k,v in buckets.items()},band_class_counts={str(k):dict(v) for k,v in types.items()},
                   seconds=round(time.time()-started,1),band_threshold=threshold,association=association,
                   box_matching='class-aware box IoU/center Hungarian matching; no GT identities used for inference',
                   max_band_candidates=0,split='val',charge_classes={str(k):dict(v) for k,v in charge_classes.items()},
                   charge_localized_queries=charge_localized,
                   charge_localization_recall=charge_localized/max(1,int(charge_counts[2])),charge_association=charge_association)
    write(output/f'{tag}_metrics.json',summary);write(output/f'{tag}_samples.json',details)
    print('VALIDATION',tag,json.dumps(summary),flush=True)
    return summary


def cpu_state(module):return {k:v.detach().cpu() for k,v in module.state_dict().items()}


def selection_score(metrics,mode,baseline=None):
    if mode=='bond':return metrics['bond_graph']['f1']
    if baseline and (metrics['atom_f1']<baseline['atom_f1']-.02 or
                     metrics['bond_graph']['f1']<baseline['bond_graph']['f1']-.02):return -1.
    return .30*metrics['atom_f1']+.35*metrics['bond_graph']['f1']+.35*metrics['charge_f1']


def checkpoint(model,optimizer,scheduler,epoch,step,best,metrics,args,path):
    state=dict(schema='independent_atom_band_joint_v1',base=cpu_state(model.base),band=cpu_state(model.band),
               optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,step=step,
               best_score=best,metrics=metrics,threshold=args.threshold,
               config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()))
    temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--init-run',type=Path,default=ROOT/'runs/band_pilot_20261113')
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--threshold',type=float,default=.2)
    p.add_argument('--seed',type=int,default=20261113);p.add_argument('--resume',type=Path)
    p.add_argument('--train-limit',type=int,default=0);p.add_argument('--val-limit',type=int,default=0)
    p.add_argument('--max-steps',type=int,default=0);p.add_argument('--skip-baseline',action='store_true')
    p.add_argument('--alignment-report',type=Path,required=True)
    p.add_argument('--init-joint',type=Path,help='Warm-start all model weights; create a fresh optimizer/schedule.')
    p.add_argument('--charge-aware',action='store_true')
    p.add_argument('--box-lr',type=float,default=2e-5);p.add_argument('--band-lr',type=float,default=1e-4)
    p.add_argument('--backbone-lr',type=float,default=2e-6);p.add_argument('--encoder-lr',type=float,default=5e-6)
    p.add_argument('--freeze-feature-epochs',type=int,default=0)
    p.add_argument('--selection',choices=['bond','balanced'],default='bond')
    p.add_argument('--association',choices=['kdtree','band_directional','geometry'],default='band_directional')
    p.add_argument('--charge-association',choices=['legacy','joint_valence'],default='joint_valence')
    p.add_argument('--patience',type=int,default=0);p.add_argument('--min-epochs',type=int,default=1)
    args=p.parse_args()
    alignment=json.loads(args.alignment_report.read_text());assert alignment['status']=='passed'
    args.output.mkdir(parents=True,exist_ok=bool(args.resume))
    # Freeze the actual CSV sampling lists for this run. Do not deduplicate
    # repeated charge entries, and do not apply a second weighted sampler.
    train=MixedIndigoDataset(args.dataset,'train',augment=True,limit=args.train_limit)
    val=MixedIndigoDataset(args.dataset,'val',augment=False,limit=args.val_limit)
    split_hashes={s:sha(d.manifest) for s,d in [('train',train),('val',val)]}
    if 'manifest_sha256' in alignment:
        assert alignment['manifest_sha256']==split_hashes,'Alignment audit must match current sampling lists'
    else:
        assert alignment['samples']==len(MixedIndigoDataset(args.dataset,'train'))+len(MixedIndigoDataset(args.dataset,'val'))
    for split,dataset in [('train',train),('val',val)]:
        snapshot=args.output/(split+dataset.manifest.suffix)
        if not args.resume:shutil.copyfile(dataset.manifest,snapshot)
        assert sha(snapshot)==split_hashes[split],'Resume manifest changed'
        dataset.manifest=snapshot
    torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    device=torch.device('cuda:0')
    prior=json.loads(((args.init_joint.parent if args.init_joint else args.init_run)/'config.json').read_text());basepath=Path(prior['base'])
    base=load_model(basepath,device).base
    band=BandDecoder().to(device)
    if args.init_joint:
        previous=torch.load(args.init_joint,map_location='cpu',weights_only=False)
        base.load_state_dict(previous['base'],strict=True)
    else:
        base.box_decoder.load_state_dict(torch.load(args.init_run/'atom_decoder.pth',map_location=device,weights_only=True),strict=True)
        previous=torch.load(args.init_run/'best_band.pth',map_location=device,weights_only=False)
    band.load_state_dict(previous['band'],strict=True)
    # The two trained heads are loaded before any baseline, smoke or optimizer update.
    assert all(torch.equal(v.cpu(),previous['band'][k].cpu()) for k,v in band.state_dict().items())
    base.requires_grad_(True);base.line_decoder.requires_grad_(False)
    model=JointModel(base,band).to(device)
    cfg=YAMLConfig(str(ROOT/'config/train_mp123_full.yml'));criterion=cfg.criterion.box_criterion.to(device).train()
    if args.charge_aware:criterion=ChargeAwareCriterion(criterion)
    training=loader(train,args.batch_size,args.workers,True);validation=loader(val,args.batch_size,args.workers)
    total_steps=len(training)*args.epochs
    optimizer=torch.optim.AdamW([dict(params=base.backbone.parameters(),lr=args.backbone_lr,name='backbone'),
                                 dict(params=base.encoder.parameters(),lr=args.encoder_lr,name='encoder'),
                                 dict(params=base.box_decoder.parameters(),lr=args.box_lr,name='atom_charge_box'),
                                 dict(params=band.parameters(),lr=args.band_lr,name='band_region')],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*min(step,total_steps)/max(1,total_steps)))))
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    code_snapshot=args.output/'source';code_snapshot.mkdir(exist_ok=True)
    code_hashes={}
    for relative in ['band_training/continue_indigo.py','band_training/charge_loss.py','band_training/charge_association.py','band_training/mixed_data.py',
                     'band_training/model.py','band_training/geometry.py','config/train_mp123_full.yml']:
        source=ROOT/relative;destination=code_snapshot/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not args.resume:shutil.copyfile(source,destination)
        code_hashes[relative]=sha(destination)
    config['code_sha256']=code_hashes
    if args.charge_aware:config['charge_loss_version']='v2_focal_alpha05_hardneg20'
    config.update(train_sampling_rows=len(train),unique_training_assets=len({r.get('asset_sample_id',r['image']) for r in train.records}),
                  charged_sampling_rows=sum(bool(r.get('has_charge',False)) for r in train.records),
                  selection_rule='0.30 atom F1 + 0.35 bond F1 + 0.35 charge F1; atom/bond each within 0.02 of initial baseline' if args.selection=='balanced' else 'bond F1')
    cache_report=args.dataset.parent/'cache_report.json'
    if cache_report.is_file():
        cache_info=json.loads(cache_report.read_text());config['source_dataset']=cache_info['source']
        write(args.output/'cache_report.json',cache_info)
    write(args.output/'alignment_report.json',alignment)
    config.update(base=str(basepath),architecture='dense_band_quad_v1',checkpoint_schema='independent_atom_band_joint_v1',
                  preserve_rgb=True,resize='isotropic letterbox 640; actual rounded scale_xy and shared padding',
                  association_loss=False,loss_weights=dict(atom_charge=1,band=2),max_band_candidates=0,
                  charge_auxiliary_weights=dict(cls=2,relative_box=1,giou=2,hard_negative=1) if args.charge_aware else None,
                  charge_focal_alpha=.5 if args.charge_aware else None,
                  sampling='snapshot CSV, ordinary shuffle; explicit repeats preserved',
                  init_weights={name:dict(path=str(path),sha256=sha(path)) for name,path in ({'joint':args.init_joint} if args.init_joint else {
                      'base':basepath,'atom_charge':args.init_run/'atom_decoder.pth','band':args.init_run/'best_band.pth'}).items()},
                  split_hashes=split_hashes)
    write(args.output/'config.json',config)
    start_epoch=0;step=0;best=-1;history=[];started=time.time();baseline=None;stale=0
    if args.resume:
        saved=torch.load(args.resume,map_location=device,weights_only=False)
        model.base.load_state_dict(saved['base']);model.band.load_state_dict(saved['band'])
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start_epoch=saved['epoch'];step=saved['step'];best=saved['best_score']
        if (args.output/'baseline_metrics.json').is_file():baseline=json.loads((args.output/'baseline_metrics.json').read_text())
        if (args.output/'progress.json').is_file():
            progress=json.loads((args.output/'progress.json').read_text());history=progress['history'];stale=progress.get('stale_epochs',0)
        if 'rng' in saved:
            torch.set_rng_state(saved['rng']['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['rng']['cuda']]);np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python'])
    print('LOADED latest atom/charge and band checkpoints; samples',len(train),len(val),'GPU',torch.cuda.get_device_name(),flush=True)
    if not args.skip_baseline and not args.resume:
        baseline=evaluate(model,validation,device,args.threshold,args.output,'baseline',association=args.association,charge_association=args.charge_association)
        best=selection_score(baseline,args.selection)
        checkpoint(model,optimizer,scheduler,0,0,best,baseline,args,args.output/'best_joint.pt')
    log=(args.output/'history.jsonl').open('a')
    for epoch in range(start_epoch,args.epochs):
        losses_epoch=Counter();batches=0
        features_trainable=epoch>=args.freeze_feature_epochs
        base.backbone.requires_grad_(features_trainable);base.encoder.requires_grad_(features_trainable)
        print('EPOCH',epoch+1,'feature_gradients',features_trainable,flush=True)
        for images,metas in training:
            model.train();base.backbone.eval();base.encoder.eval();base.line_decoder.eval()
            images=images.to(device,non_blocking=True)
            box_targets=[{k:v.to(device,non_blocking=True) for k,v in m['boxes'].items()} for m in metas]
            band_targets=[(m['band'][0].to(device,non_blocking=True),
                           [(l,y,x,o.to(device),q.to(device)) for l,y,x,o,q in m['band'][1]]) for m in metas]
            optimizer.zero_grad(set_to_none=True)
            boxes,bands=model(images,box_targets)
            box_losses=criterion(boxes,box_targets)
            atoms_loss=sum(box_losses.values())
            bands_loss=band_loss(bands,band_targets)
            total=atoms_loss+2*bands_loss['total']
            if not torch.isfinite(total):raise RuntimeError('Nonfinite joint loss')
            total.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            if not float(grad)>0:raise RuntimeError('Disconnected training gradients')
            optimizer.step();scheduler.step();step+=1;batches+=1
            current=dict(atom_charge=float(atoms_loss.detach()),band=float(bands_loss['total'].detach()),
                         band_class=float(bands_loss['exist_type'].detach()),region=float(bands_loss['region'].detach()),total=float(total.detach()))
            current.update(charge_instances=sum(len(m['ann']['charges']) for m in metas),
                           charged_images=sum(bool(m['ann']['charges']) for m in metas),gradient_norm=float(grad))
            current.update({k:float(v.detach()) for k,v in box_losses.items() if k.startswith('loss_charge_')})
            losses_epoch.update(current)
            if step%25==0 or step==1:
                row=dict(epoch=epoch+1,step=step,seconds=round(time.time()-started,1),lr=optimizer.param_groups[-1]['lr'],**current)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if step%500==0:checkpoint(model,optimizer,scheduler,epoch,step,best,None,args,args.output/'last_joint.pt')
            if args.max_steps and step>=args.max_steps:break
        metrics=evaluate(model,validation,device,args.threshold,args.output,f'epoch_{epoch+1:02d}',association=args.association,charge_association=args.charge_association)
        score=selection_score(metrics,args.selection,baseline)
        if score>best:
            best=score;checkpoint(model,optimizer,scheduler,epoch+1,step,best,metrics,args,args.output/'best_joint.pt')
            stale=0
        else:stale+=1
        checkpoint(model,optimizer,scheduler,epoch+1,step,best,metrics,args,args.output/'last_joint.pt')
        history.append(dict(epoch=epoch+1,step=step,train_losses={k:v/max(1,batches) for k,v in losses_epoch.items()},validation=metrics))
        write(args.output/'progress.json',dict(status='training',completed_epochs=epoch+1,step=step,
                                              best_selection_score=best,stale_epochs=stale,history=history))
        if args.max_steps and step>=args.max_steps:break
        if args.patience and epoch+1>=args.min_epochs and stale>=args.patience:
            print('EARLY_STOP',epoch+1,'stale_epochs',stale,flush=True);break
    log.close()
    best_state=torch.load(args.output/'best_joint.pt',map_location='cpu',weights_only=False)
    result=dict(status='completed',epochs_completed=epoch+1,optimizer_steps=step,
                train_images=len(train),validation_images=len(val),best_selection_score=best,
                best_bond_f1=best_state['metrics']['bond_graph']['f1'],
                best_epoch=best_state['epoch'],best_metrics=best_state['metrics'],history=history,
                seconds=round(time.time()-started,1),
                caveats=['Validation is used for checkpoint selection; no independent test claim.',
                         'Element labels outside the existing 13 atom classes are mapped to other; text recognition is not trained.',
                         'Band regions and classes are supervised independently of atom detections/indices.',
                         'Letterbox alignment audit passed before training.'])
    write(args.output/'results.json',result);print('COMPLETED',json.dumps({k:v for k,v in result.items() if k not in ['history','best_metrics']}),flush=True)
    write(args.output/'progress.json',dict(status='completed',completed_epochs=epoch+1,step=step,
                                          best_selection_score=best,history=history))

if __name__=='__main__':main()
