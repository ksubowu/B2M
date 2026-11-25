"""Charge-only auxiliary supervision for the existing independent box decoder.

Use the same one-to-one atom/charge matching as the original criterion. No
atom association or band targets are used here. GT boxes remain actual ink.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss,box_convert,generalized_box_iou


def charge_terms(output,targets,indices):
    logits=output['pred_logits'];pred_boxes=output['pred_boxes']
    positive=torch.zeros_like(logits[:,:,13:])
    prediction=[];truth=[]
    for bi,(queries,objects) in enumerate(indices):
        queries=queries.to(logits.device);objects=objects.to(logits.device)
        labels=targets[bi]['labels'][objects];mask=labels>=13
        q=queries[mask];t=objects[mask];labels=labels[mask]
        positive[bi,q,labels-13]=1
        prediction.append(pred_boxes[bi,q]);truth.append(targets[bi]['boxes'][t])
    prediction=torch.cat(prediction);truth=torch.cat(truth);count=len(truth)
    # Hard negatives are supervised even for images without charges. Positive
    # class targets are 1, independent of localization IoU (unlike VFL).
    classification=sigmoid_focal_loss(logits[:,:,13:],positive,alpha=.5,gamma=2,reduction='sum')/max(1,count)
    # Balance the enriched training prior with explicit suppression of the
    # highest scoring negative charge hypotheses (letters/bonds/background).
    # True positive class entries are excluded, so valid charge logits never
    # receive this negative penalty. Twenty negatives per image, not per GT.
    negatives=logits[:,:,13:].flatten(1).masked_fill(positive.flatten(1).bool(),-100)
    hard_negative=F.softplus(negatives.topk(min(20,negatives.shape[1]),dim=1).values).mean()
    zero=pred_boxes.sum()*0
    if not count:return dict(cls=classification,relative_box=zero,giou=zero,hard_negative=hard_negative)
    center=F.smooth_l1_loss((prediction[:,:2]-truth[:,:2])/truth[:,2:].clamp(min=4/640),
                          torch.zeros_like(truth[:,:2]),reduction='sum')/count
    log_size=torch.log(prediction[:,2:].clamp(min=1e-6)/truth[:,2:].clamp(min=1e-6))
    size=F.smooth_l1_loss(log_size,torch.zeros_like(log_size),reduction='sum')/count
    giou=generalized_box_iou(box_convert(prediction,'cxcywh','xyxy'),box_convert(truth,'cxcywh','xyxy')).diag()
    return dict(cls=classification,relative_box=center+size,giou=(1-giou).mean(),hard_negative=hard_negative)


class ChargeAwareCriterion(nn.Module):
    def __init__(self,base):super().__init__();self.base=base
    def forward(self,outputs,targets):
        losses=self.base(outputs,targets)
        stages=[(outputs,1.)]
        if outputs.get('aux_outputs'):stages.append((outputs['aux_outputs'][-1],.5))
        if outputs.get('enc_aux_outputs'):stages.append((outputs['enc_aux_outputs'][-1],.5))
        totals={};normalizer=sum(w for _,w in stages)
        for stage,weight in stages:
            indices=self.base.matcher(stage,targets)['indices']
            terms=charge_terms(stage,targets,indices)
            for key,value in terms.items():totals[key]=totals.get(key,0)+weight/normalizer*value
        for key,weight in dict(cls=2.,relative_box=1.,giou=2.,hard_negative=1.).items():
            losses['loss_charge_'+key]=weight*totals[key]
        return losses
