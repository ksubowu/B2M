"""Fixed proposals plus residual entity heads; protect correct ordinary atoms."""
import torch
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_convert,box_iou
from pair_training.retained_ocr import ResidualLinear


def reference_counts(cases,keys=None):
    result=defaultdict(lambda:dict(correct=0,total=0));seen=set()
    for row in cases:
        key=(row['domain'],row['image'])
        if keys is not None and key not in keys:continue
        if key in seen:raise ValueError('Duplicate reference case')
        seen.add(key);domain=row['domain'];size=str(row['native_size']);kind=row['label_type']
        for group in [domain,domain+':'+size,domain+':'+kind,domain+':'+size+':'+kind]:
            result[group]['total']+=1;result[group]['correct']+=bool(row['correct'])
    if keys is not None and seen!=keys:raise ValueError('Reference subset membership changed')
    return dict(result)


def install_entity_adapters(decoder,anchor,rank=4,spatial_adapter=False,full_decoder=False):
    decoder.requires_grad_(full_decoder);anchor.requires_grad_(False)
    names=[] if full_decoder else [n for n,m in decoder.named_modules() if isinstance(m,nn.Linear) and
           (n.startswith('dec_bbox_head.') or n.startswith('dec_score_head.') or
            (spatial_adapter and n.startswith('decoder.layers.') and '.self_attn.' not in n))]
    for name in names:
        parent,_,leaf=name.rpartition('.');module=decoder.get_submodule(parent)
        setattr(module,leaf,ResidualLinear(getattr(module,leaf),rank))
    for i in [0,2]:anchor.anchor_predictor[i]=ResidualLinear(anchor.anchor_predictor[i],rank)
    if not names and not full_decoder:raise ValueError('Unsupported decoder: no entity heads')
    return names


def matched_teacher_loss(student,teacher,targets,matcher):
    """Object correspondence remains valid when proposal/query order changes."""
    si=matcher({k:student[k] for k in ['pred_logits','pred_boxes']},targets)['indices']
    ti=matcher(teacher,targets)['indices'];sb=[];tb=[];sl=[];tl=[]
    for batch,((sq,sg),(tq,tg)) in enumerate(zip(si,ti)):
        device=student['pred_boxes'].device
        sq,sg,tq,tg=[x.to(device) for x in [sq,sg,tq,tg]]
        if not len(tq):continue
        student_by_gt=torch.full((len(targets[batch]['labels']),),-1,dtype=torch.long,device=device)
        student_by_gt[sg]=sq
        boxes=teacher['pred_boxes'][batch,tq];target=targets[batch]['boxes'][tg]
        xy=box_convert(boxes,'cxcywh','xyxy');gt=box_convert(target,'cxcywh','xyxy')
        overlap=box_iou(xy,gt).diag()
        score,label=teacher['pred_logits'][batch,tq,:13].sigmoid().max(-1)
        keep=(student_by_gt[tg]>=0)&(label==targets[batch]['labels'][tg])&(score>=.4)&(overlap>=.5)
        intersection=(torch.minimum(xy[:,2:],gt[:,2:])-torch.maximum(xy[:,:2],gt[:,:2])).clamp_min(0).prod(-1)
        complete=(score>=.8)&(overlap>=.8)&(intersection/target[:,2:].prod(-1).clamp_min(1e-8)>=.95)
        keep=keep&((label!=0)|complete)
        if not keep.any():continue
        student_q=student_by_gt[tg[keep]]
        sb.extend(student['pred_boxes'][batch,student_q].unbind());tb.extend(boxes[keep].unbind())
        sl.extend(student['pred_logits'][batch,student_q].unbind());tl.extend(teacher['pred_logits'][batch,tq[keep]].unbind())
    if not sb:return student['pred_boxes'].sum()*0,dict(protected=0)
    boxes=torch.stack(tb).detach();logits=torch.stack(sl);prior=torch.stack(tl).detach();p=prior.sigmoid()
    cls=(F.binary_cross_entropy_with_logits(logits,p,reduction='none')-F.binary_cross_entropy_with_logits(prior,p,reduction='none')).sum(-1).mean().clamp_min(0)
    geometry=F.smooth_l1_loss((torch.stack(sb)-boxes)/boxes[:,2:].clamp_min(.003).repeat(1,2),torch.zeros_like(boxes))
    return cls+geometry,dict(protected=len(sb),classification=float(cls.detach()),geometry=float(geometry.detach()),matched_by_gt=True)


def ordinary_teacher_loss(student,teacher,targets,protect_complete_text=False):
    """Fixed proposal heads retain query correspondence. Ignore teacher errors.

    Supervised GT loss remains responsible for correcting errors and full text
    labels. This penalty protects only confident, GT-matched ordinary atoms.
    """
    scores,labels=teacher['pred_logits'][...,:13].sigmoid().max(-1)
    masks=[]
    for i,target in enumerate(targets):
        if not len(target['boxes']):
            masks.append(torch.zeros_like(labels[i],dtype=torch.bool));continue
        xy=box_convert(teacher['pred_boxes'][i],'cxcywh','xyxy');gtxy=box_convert(target['boxes'],'cxcywh','xyxy')
        overlap=box_iou(xy,gtxy)
        correct=((labels[i,:,None]==target['labels'][None,:]) & (overlap>=.5)).any(-1)
        keep=correct & (scores[i]>=.4) & (labels[i]!=0)
        if protect_complete_text:
            inter=(torch.minimum(xy[:,None,2:],gtxy[None,:,2:])-torch.maximum(xy[:,None,:2],gtxy[None,:,:2])).clamp_min(0).prod(-1)
            coverage=inter/(target['boxes'][:,2:].prod(-1)[None,:].clamp_min(1e-8))
            whole=((overlap>=.8)&(coverage>=.95)&(target['labels'][None,:]==0)).any(-1)
            keep=keep|((labels[i]==0)&(scores[i]>=.8)&whole)
        masks.append(keep)
    keep=torch.stack(masks)
    if not keep.any():return student['pred_boxes'].sum()*0,dict(protected=0)
    prior=teacher['pred_logits'][keep].detach();logits=student['pred_logits'][keep]
    p=prior.sigmoid()
    divergence=(F.binary_cross_entropy_with_logits(logits,p,reduction='none')-
                F.binary_cross_entropy_with_logits(prior,p,reduction='none'))
    classification=(divergence.sum(-1).mean() if protect_complete_text else divergence.mean()).clamp_min(0)
    # Normalize by teacher box size: a pixel shift of a small atom matters.
    boxes=teacher['pred_boxes'][keep].detach()
    span=boxes[:,2:].clamp_min(.003).repeat(1,2)
    geometry=F.smooth_l1_loss((student['pred_boxes'][keep]-boxes)/span,torch.zeros_like(boxes))
    return classification+geometry,dict(protected=int(keep.sum()),classification=float(classification.detach()),geometry=float(geometry.detach()))


def focused_text_loss(outputs,targets,indices=None):
    """Explicit full-label supervision and suppression of contained fragments.

    All letters of a label have exactly one wildcard target. Genuine ordinary
    atom centers remain exempt even if a text box overlaps them.
    """
    from scipy.optimize import linear_sum_assignment
    from torchvision.ops import generalized_box_iou
    total=outputs['pred_boxes'].sum()*0;regions=0;fragments=0
    for batch,(logits,boxes,target) in enumerate(zip(outputs['pred_logits'],outputs['pred_boxes'],targets)):
        text=target['boxes'][target['labels']==0]
        if not len(text):continue
        xy=box_convert(boxes,'cxcywh','xyxy');gtxy=box_convert(text,'cxcywh','xyxy')
        if indices is None:
            # Standalone diagnostic use; production training passes the SAME
            # global Hungarian assignment used by the detector criterion.
            probability=logits[:,0].sigmoid().clamp(1e-6,1-1e-6)
            cls=.25*(1-probability).square()*(-probability.log())-.75*probability.square()*(-(1-probability).log())
            cost=5*torch.cdist(boxes,text,p=1)-2*generalized_box_iou(xy,gtxy)+2*cls[:,None]
            ii,jj=linear_sum_assignment(cost.detach().cpu().numpy())
            qi=torch.as_tensor(ii,device=boxes.device);gt=text[torch.as_tensor(jj,device=boxes.device)];all_matched=qi
        else:
            ii,jj=[v.to(boxes.device) for v in indices[batch]]
            selected=target['labels'][jj]==0;qi=ii[selected];gt=target['boxes'][jj[selected]];all_matched=ii
        if len(qi):
            # Quality-aware VFL already supervises classification. A second
            # hard-positive BCE would conflict with its IoU-calibrated score.
            total=total+5*F.l1_loss(boxes[qi],gt)+(1-generalized_box_iou(xy[qi],box_convert(gt,'cxcywh','xyxy')).diag()).mean()
        regions+=len(text)
        # Suppress little letter boxes only when contained in a complete GT
        # label and not occupying a real ordinary-atom location.
        centers=boxes[:,:2];inside=((centers[:,None,:]>=gtxy[None,:,:2])&(centers[:,None,:]<=gtxy[None,:,2:])).all(-1)
        small=boxes[:,2:].prod(-1)[:,None]<1.5*text[:,2:].prod(-1)[None,:]
        inter=(torch.minimum(xy[:,None,2:],gtxy[None,:,2:])-torch.maximum(xy[:,None,:2],gtxy[None,:,:2])).clamp_min(0).prod(-1)
        contained=inter/boxes[:,2:].prod(-1)[:,None].clamp_min(1e-8)>=.5
        mask=(inside&small&contained).any(-1);mask[all_matched]=False
        ordinary=target['boxes'][target['labels']!=0]
        if len(ordinary):
            # Query boxes that cover a real atom center are never treated as
            # fragmented text, independently of the predicted class.
            genuine=((ordinary[None,:,:2]>=xy[:,None,:2])&(ordinary[None,:,:2]<=xy[:,None,2:])).all(-1).any(-1)
            mask=mask&~genuine
        if mask.any():
            total=total+F.softplus(logits[mask,:13].max(-1).values).mean();fragments+=int(mask.sum())
    return total/max(1,len(targets)),dict(text_regions=regions,fragment_queries=fragments)


def frozen_state_unchanged(module,original):
    now=module.state_dict()
    adapters={n for n,m in module.named_modules() if isinstance(m,ResidualLinear)}
    for name,value in original.items():
        prefix,_,leaf=name.rpartition('.')
        key=prefix+'.base.'+leaf if prefix in adapters else name
        if not torch.equal(value.cpu(),now[key].detach().cpu()):return False
    return True
