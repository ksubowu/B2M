"""Two bond heads sharing spatial instance queries, with independent scores.

Line selection never reads band scores. Each head has its own objectness and
geometry. Band type is categorical (one type per instance). Atom IDs/boxes are
not inputs to the decoder or bond criterion.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .model import BandDecoder,targets_for


class LineHead(nn.Module):
    def __init__(self):
        super().__init__();self.adapt=nn.Conv2d(96,96,3,padding=1)
        nn.init.dirac_(self.adapt.weight);nn.init.zeros_(self.adapt.bias)
        self.score=nn.Conv2d(96,1,1);self.offset=nn.Conv2d(96,2,1);self.endpoints=nn.Conv2d(96,4,1)
        nn.init.constant_(self.score.bias,-2.19)

    def forward(self,x):
        x=self.adapt(x)
        return dict(score_logits=self.score(x),offset=self.offset(x),endpoints=self.endpoints(x))


class BandHead(nn.Module):
    def __init__(self):
        super().__init__();self.score=nn.Conv2d(96,1,1);self.types=nn.Conv2d(96,6,1)
        self.offset=nn.Conv2d(96,2,1);self.quad=nn.Conv2d(96,48,1)
        nn.init.constant_(self.score.bias,-2.19)

    def forward(self,x):
        return dict(score_logits=self.score(x),type_logits=self.types(x),offset=self.offset(x),quad=self.quad(x))


class ScoredBondDecoder(nn.Module):
    def __init__(self):
        super().__init__();shared=BandDecoder()
        self.project=shared.project;self.image=shared.image;self.body=shared.body
        self.line_head=LineHead();self.band_head=BandHead()

    def initialize_from_paired_v2(self,state):
        for name in ['project','image','body']:
            prefix=name+'.';getattr(self,name).load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
        with torch.no_grad():
            for suffix in ['weight','bias']:
                getattr(self.band_head.types,suffix).copy_(state['heat.'+suffix])
                getattr(self.band_head.quad,suffix).copy_(state['quad.'+suffix])
                getattr(self.band_head.offset,suffix).copy_(state['offset.'+suffix])
                getattr(self.line_head.offset,suffix).copy_(state['offset.'+suffix])
                getattr(self.line_head.endpoints,suffix).copy_(state['line.'+suffix][:4])
                # Fresh objectness heads warm-start from the majority single
                # bond channel. They are separate parameters, not aliased.
                getattr(self.line_head.score,suffix).copy_(state['heat.'+suffix][:1])
                getattr(self.band_head.score,suffix).copy_(state['heat.'+suffix][:1])

    def forward(self,features,gray):
        pixels=self.image(gray)
        f=F.interpolate(self.project(features[0]),size=pixels.shape[-2:],mode='bilinear',align_corners=False)
        x=self.body(torch.cat((f,pixels),1))
        return dict(line=self.line_head(x),band=self.band_head(x))


def scored_targets(labels,quads,lines,device='cpu',size=160):
    if not len(labels)==len(quads)==len(lines):raise ValueError('Unpaired bond annotations')
    # Both branches use the same instance query. Class-agnostic heatmaps need
    # separated anchors across ALL types, not just within a bond class.
    heat,base=targets_for([0]*len(labels),quads,device,size=size,resolve_collisions=True)
    if len(base)!=len(labels):raise ValueError('Never drop a line/band target')
    entries=[]
    for label,(_,y,x,offset,quad),line in zip(labels,base,lines):
        center=(torch.tensor([x,y],device=device)+offset)*4
        line=(torch.as_tensor(line,device=device,dtype=torch.float32)-center).flatten()/32
        entries.append((int(label),y,x,offset,quad,line))
    return heat[:1],entries


def move_scored_targets(targets,device):
    return [(heat.to(device),[(label,y,x,off.to(device),quad.to(device),line.to(device)) for label,y,x,off,quad,line in entries]) for heat,entries in targets]


def focal_score(logits,heat):
    prob=logits.float().sigmoid().clamp(1e-5,1-1e-5);positive=heat.eq(1)
    return (-(1-prob).square()*prob.log()*positive-prob.square()*(1-prob).log()*(1-heat).pow(4)*(~positive)).sum()/positive.sum().clamp_min(1)


def scored_bond_loss(output,targets):
    heat=torch.stack([h for h,_ in targets]);line=output['line'];band=output['band']
    result=dict(line_score=focal_score(line['score_logits'],heat),band_score=focal_score(band['score_logits'],heat))
    indices=[];offsets=[];quads=[];endpoints=[]
    for bi,(_,entries) in enumerate(targets):
        for label,y,x,offset,quad,segment in entries:
            indices.append((bi,label,y,x));offsets.append(offset);quads.append(quad);endpoints.append(segment)
    if indices:
        bi,label,y,x=torch.tensor(indices,device=heat.device).T
        target_offset=torch.stack(offsets)
        result['line_center']=F.smooth_l1_loss(line['offset'][bi,:,y,x].float(),target_offset)
        result['band_center']=F.smooth_l1_loss(band['offset'][bi,:,y,x].float(),target_offset)
        result['line_endpoints']=F.smooth_l1_loss(line['endpoints'][bi,:,y,x].float(),torch.stack(endpoints))
        result['band_type']=F.cross_entropy(band['type_logits'][bi,:,y,x].float(),label)
        values=band['quad'].reshape(band['quad'].shape[0],6,8,*band['quad'].shape[-2:])
        result['band_region']=F.smooth_l1_loss(values[bi,label,:,y,x].float(),torch.stack(quads))
    else:
        for name,tensor in [('line_center',line['offset']),('line_endpoints',line['endpoints']),('band_center',band['offset']),('band_type',band['type_logits']),('band_region',band['quad'])]:
            result[name]=tensor.sum()*0
    result['line_total']=result['line_score']+result['line_center']+5*result['line_endpoints']
    result['band_total']=result['band_score']+result['band_type']+result['band_center']+result['band_region']
    result['total']=result['line_total']+result['band_total']
    return result


@torch.no_grad()
def decode_scored_bonds(output,index=0,line_threshold=.2,band_threshold=.2,apply_line_peaks=True):
    """Independent thresholding, with transparent shared-query pairing.

    The query is (x,y) on the feature grid. For every accepted line query we
    also inspect the band head at that SAME query (not requiring a second peak).
    Standalone band peaks are independent of line selection. Additional dense
    band reads at line queries are reported as pairing probes, not new band
    detections. There is no band->line gate.
    """
    lo=output['line'];bo=output['band'];lp=lo['score_logits'][index,0].float().sigmoid();bp=bo['score_logits'][index,0].float().sigmoid()
    lpeak=lp==F.max_pool2d(lp[None,None],3,1,1)[0,0]
    bpeak=bp==F.max_pool2d(bp[None,None],3,1,1)[0,0]
    selected_lines=(lpeak if apply_line_peaks else torch.ones_like(lpeak))&(lp>=line_threshold)
    ly,lx=torch.where(selected_lines)
    lines=[];bands=[];probes=[];_,h,w=bo['type_logits'][index].shape
    peak_queries=set(torch.where(lpeak.flatten())[0].tolist())
    if len(lx):
        center=(torch.stack((lx,ly),1).float()+lo['offset'][index,:,ly,lx].T.float())*4
        ends=lo['endpoints'][index,:,ly,lx].T.reshape(-1,2,2).float()*32+center[:,None]
        for query,score,line in zip((ly*w+lx).tolist(),lp[ly,lx].tolist(),ends.cpu().numpy()):
            lines.append(dict(query_id=query,line_score=score,line=line,is_local_peak=query in peak_queries))
    by,bx=torch.where(((bpeak&(bp>=band_threshold))|selected_lines)&(bp>=band_threshold))
    if len(bx):
        probability=bo['type_logits'][index,:,by,bx].T.float().softmax(-1);type_score,label=probability.max(-1)
        confidence=bp[by,bx]*type_score;keep=confidence>=band_threshold
        by,bx,label,type_score,confidence=by[keep],bx[keep],label[keep],type_score[keep],confidence[keep]
        center=(torch.stack((bx,by),1).float()+bo['offset'][index,:,by,bx].T.float())*4
        quad=bo['quad'][index].reshape(6,8,h,w)[label,:,by,bx].reshape(-1,4,2).float()*32+center[:,None]
        values=zip((by*w+bx).tolist(),confidence.tolist(),bp[by,bx].tolist(),type_score.tolist(),label.tolist(),quad.cpu().numpy(),bpeak[by,bx].tolist())
        for query,score,presence,ts,cls,q,standalone in values:
            record=dict(query_id=query,band_score=score,band_presence_score=presence,type_score=ts,score=score,label=cls,quad=q,
                candidate_source='band_peak' if standalone else 'same_query_pairing_probe')
            (bands if standalone else probes).append(record)
    return dict(lines=sorted(lines,key=lambda r:-r['line_score']),bands=sorted(bands,key=lambda r:-r['score']),
        paired_band_probes=sorted(probes,key=lambda r:-r['score']),line_threshold=line_threshold,band_threshold=band_threshold,grid_width=w)
