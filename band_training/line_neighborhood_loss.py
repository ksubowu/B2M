"""Ordered geometry supervision around shared queries; no atom IDs or boxes."""
import torch
from torch.nn import functional as F


def line_neighborhood_terms(head,targets):
    entries=[];offset=[];segment=[]
    for bi,(_,records) in enumerate(targets):
        for _,y,x,o,_,line in records:
            entries.append((bi,y,x));offset.append(o);segment.append(line)
    zero=head['endpoints'].sum()*0
    if not entries:return dict(neighborhood=zero,relative=zero,direction=zero)
    device=head['endpoints'].device
    bi,y,x=torch.tensor(entries,device=device).T;off=torch.stack(offset);seg=torch.stack(segment).reshape(-1,2,2)
    anchor=(torch.stack((x,y),1)+off)*4;truth=seg*32+anchor[:,None]
    length=torch.linalg.vector_norm(truth[:,1]-truth[:,0],dim=-1).clamp_min(12.)
    center=(torch.stack((x,y),1)+head['offset'][bi,:,y,x].float())*4
    pred=head['endpoints'][bi,:,y,x].float().reshape(-1,2,2)*32+center[:,None]
    relative=F.smooth_l1_loss((pred-truth)/length[:,None,None],torch.zeros_like(pred))
    direction=(1-F.cosine_similarity(pred[:,1]-pred[:,0],truth[:,1]-truth[:,0],dim=-1)).mean()
    delta=torch.tensor([(dy,dx) for dy in [-1,0,1] for dx in [-1,0,1]],device=device)
    ny=(y[:,None]+delta[:,0]).flatten();nx=(x[:,None]+delta[:,1]).flatten()
    batch=bi[:,None].expand(-1,9).flatten();owner=torch.arange(len(entries),device=device)[:,None].expand(-1,9).flatten()
    h,w=head['endpoints'].shape[-2:];valid=(ny>=0)&(ny<h)&(nx>=0)&(nx<w)
    ny,nx,batch,owner=ny[valid],nx[valid],batch[valid],owner[valid]
    key=batch*h*w+ny*w+nx;_,inverse,counts=key.unique(return_inverse=True,return_counts=True)
    # Overlapping target neighborhoods are ambiguous: do not ask one cell to
    # represent two different lines, even when the two bonds share an atom.
    unique=counts[inverse]==1;ny,nx,batch,owner=ny[unique],nx[unique],batch[unique],owner[unique]
    if len(owner):
        center=(torch.stack((nx,ny),1)+head['offset'][batch,:,ny,nx].float())*4
        p=head['endpoints'][batch,:,ny,nx].float().reshape(-1,2,2)*32+center[:,None]
        distance=(ny-y[owner]).square()+(nx-x[owner]).square();weight=torch.exp(-.5*distance.float())
        error=F.smooth_l1_loss((p-truth[owner])/length[owner,None,None],torch.zeros_like(p),reduction='none').mean((1,2))
        neighborhood=(weight*error).sum()/weight.sum().clamp_min(1.)
    else:neighborhood=zero
    return dict(neighborhood=neighborhood,relative=relative,direction=direction)
