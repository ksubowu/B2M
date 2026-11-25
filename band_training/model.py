"""Dense visual band decoder: class centers + ordered tapered quadrilateral regions.

No atom detections, atom indices, pair candidates, or association targets enter
this module or its criterion. The quadrilateral represents a region, not a
centerline; its four vertices are regressed in fixed order without swapping.
"""
import torch
from torch import nn
from torch.nn import functional as F


class BandDecoder(nn.Module):
    def __init__(self, classes=6):
        super().__init__()
        self.project=nn.Sequential(nn.Conv2d(256,64,1),nn.GroupNorm(8,64),nn.SiLU())
        self.image=nn.Sequential(nn.Conv2d(1,16,5,2,2),nn.SiLU(),nn.Conv2d(16,32,3,2,1),nn.SiLU())
        self.body=nn.Sequential(nn.Conv2d(96,96,3,padding=1),nn.GroupNorm(8,96),nn.SiLU(),
                                nn.Conv2d(96,96,3,padding=1),nn.GroupNorm(8,96),nn.SiLU())
        self.heat=nn.Conv2d(96,classes,1)
        self.offset=nn.Conv2d(96,2,1)
        self.quad=nn.Conv2d(96,classes*8,1)
        nn.init.constant_(self.heat.bias,-2.19)
        nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.quad.weight);nn.init.zeros_(self.quad.bias)

    def hidden(self,features,gray):
        pixels=self.image(gray)
        feature=F.interpolate(self.project(features[0]),size=pixels.shape[-2:],mode='bilinear',align_corners=False)
        return self.body(torch.cat((feature,pixels),1))

    def forward(self,features,gray):
        x=self.hidden(features,gray)
        return {'heat':self.heat(x),'offset':self.offset(x),'quad':self.quad(x)}


def targets_for(labels,quads,device,size=160,resolve_collisions=False):
    """Build geometry-only targets; optional displaced anchors retain crowded bonds.

    The decoder's offset recovers the exact original region center, even when
    its heatmap anchor moves to a nearby free cell. No atom data is involved.
    Same-class anchors are separated for 3x3 peak NMS, and all classes use
    distinct cells because the center-offset head is shared across classes.
    """
    heat=torch.zeros((6,size,size),device=device)
    coords=torch.arange(-2,3,device=device,dtype=torch.float32)
    kernel=torch.exp(-(coords[:,None].square()+coords[None,:].square())/2)
    entries=[];seen=set()
    occupied=set();anchors=[]
    natural={tuple((torch.as_tensor(q,dtype=torch.float32).mean(0)/4).floor().long().tolist()) for q in quads}
    for label,quad in zip(labels,quads):
        q=torch.as_tensor(quad,device=device,dtype=torch.float32)
        center=q.mean(0)/4
        xi,yi=center.floor().long().tolist()
        if not (0<=xi<size and 0<=yi<size):continue
        if resolve_collisions:
            def available(x,y):
                return (0<=x<size and 0<=y<size and (x,y) not in occupied and
                        all(c!=label or max(abs(x-ax),abs(y-ay))>1 for c,ax,ay in anchors))
            if not available(xi,yi):
                choices=[]
                for radius in range(1,9):
                    for dy in range(-radius,radius+1):
                        for dx in range(-radius,radius+1):
                            x,y=xi+dx,yi+dy
                            if max(abs(dx),abs(dy))!=radius or not available(x,y):continue
                            if (x,y) in natural:continue
                            choices.append((dx*dx+dy*dy,y,x))
                    if choices:break
                if not choices:raise ValueError('No free anchor for crowded band; do not discard target')
                _,yi,xi=min(choices)
            occupied.add((xi,yi));anchors.append((label,xi,yi))
        elif (label,xi,yi) in seen:raise ValueError('Same-class center collision')
        seen.add((label,xi,yi))
        x0,x1=max(0,xi-2),min(size,xi+3)
        y0,y1=max(0,yi-2),min(size,yi+3)
        heat[label,y0:y1,x0:x1]=torch.maximum(heat[label,y0:y1,x0:x1],
            kernel[y0-yi+2:y1-yi+2,x0-xi+2:x1-xi+2])
        entries.append((label,yi,xi,center-torch.tensor([xi,yi],device=device),
                        (q-center[None]*4).flatten()/32))
    return heat,entries


def band_loss(output,targets):
    # Only geometric region targets and type labels are accessible here.
    heat=torch.stack([t[0] for t in targets])
    prob=output['heat'].sigmoid().clamp(1e-5,1-1e-5)
    pos=heat.eq(1);neg=~pos
    n=pos.sum().clamp_min(1)
    focal=(-(1-prob).square()*prob.log()*pos-prob.square()*(1-prob).log()*(1-heat).pow(4)*neg).sum()/n
    indices=[];offsets=[];geometries=[]
    for bi,(_,entries) in enumerate(targets):
        for label,yi,xi,off,quad in entries:
            indices.append((bi,label,yi,xi));offsets.append(off);geometries.append(quad)
    if indices:
        bi,label,yi,xi=torch.tensor(indices,device=prob.device).T
        off=F.smooth_l1_loss(output['offset'][bi,:,yi,xi].float(),torch.stack(offsets))
        geometry=output['quad'].reshape(output['quad'].shape[0],6,8,*output['quad'].shape[-2:])
        geom=F.smooth_l1_loss(geometry[bi,label,:,yi,xi].float(),torch.stack(geometries))
    else:
        off=output['offset'].sum()*0;geom=output['quad'].sum()*0
    return {'exist_type':focal,'center':off,'region':geom,'total':focal+off+5*geom}


@torch.no_grad()
def decode(output,index=0,minimum=.05,topk=None):
    """Keep all thresholded local peaks by default; topk>0 is an optional cap.

    The pilot's previous 160 limit was a runtime safeguard, not model capacity.
    Thresholding before sorting and batched geometry extraction avoid sorting
    all background locations or synchronizing the GPU once per detection.
    """
    if topk is not None and topk<0:raise ValueError('topk must be nonnegative or None')
    heat=output['heat'][index].sigmoid()
    peaks=heat==F.max_pool2d(heat[None],3,1,1)[0]
    indices=torch.where((peaks & (heat>=minimum)).flatten())[0]
    if not len(indices):return []
    scores=heat.flatten()[indices]
    order=scores.argsort(descending=True,stable=True)
    if topk:order=order[:topk]
    indices=indices[order];scores=scores[order]
    _,h,w=heat.shape
    labels=indices//(h*w);ys=(indices//w)%h;xs=indices%w
    centers=(torch.stack((xs,ys),1).float()+output['offset'][index,:,ys,xs].T)*4
    geometry=output['quad'][index].reshape(6,8,h,w)
    quads=geometry[labels,:,ys,xs].reshape(-1,4,2).float()*32+centers[:,None,:]
    result=[{'label':label,'score':score,'quad':quad}
            for label,score,quad in zip(labels.tolist(),scores.tolist(),quads.cpu().numpy())]
    if 'line' in output:
        line_geometry=output['line'][index].reshape(6,4,h,w)
        lines=line_geometry[labels,:,ys,xs].reshape(-1,2,2).float()*32+centers[:,None,:]
        for item,line in zip(result,lines.cpu().numpy()):item['line']=line
    return result
