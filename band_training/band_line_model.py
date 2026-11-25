"""Paired bond outputs without atom queries, indices or pointer supervision."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import BandDecoder,band_loss


class BandLineDecoder(BandDecoder):
    def __init__(self,classes=6):
        super().__init__(classes)
        self.line=nn.Conv2d(96,classes*4,1)

    def initialize_from_band(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if set(missing)!={'line.weight','line.bias'} or unexpected:
            raise ValueError((missing,unexpected))
        # Same ordered ends as the previous quad, extended to atom anchors by
        # new line supervision. Do not discard the trained bond representation.
        with torch.no_grad():
            self.line.weight.copy_(self.quad.weight.reshape(6,2,2,2,96,1,1).mean(2).reshape(24,96,1,1))
            self.line.bias.copy_(self.quad.bias.reshape(6,2,2,2).mean(2).reshape(24))

    def forward(self,features,gray):
        x=self.hidden(features,gray)
        return dict(heat=self.heat(x),offset=self.offset(x),quad=self.quad(x),line=self.line(x))


def paired_bond_loss(output,targets,line_targets,line_weight=5.,region_weight=1.):
    """Shared detection anchor; fixed-order coordinate loss, never end swapping.

    band targets contain only class and region. line_targets are paired 2x2
    coordinates, without atom IDs or any atom detector output.
    """
    losses=band_loss(output,targets)
    indices=[];truth=[]
    for bi,((_,entries),lines) in enumerate(zip(targets,line_targets)):
        if len(entries)!=len(lines):raise ValueError('Band/line count mismatch')
        for (label,y,x,offset,quad),line in zip(entries,lines):
            center=(torch.as_tensor([x,y],device=offset.device)+offset)*4
            indices.append((bi,label,y,x))
            truth.append((torch.as_tensor(line,device=offset.device,dtype=torch.float32)-center).flatten()/32)
    if indices:
        bi,label,y,x=torch.tensor(indices,device=output['line'].device).T
        values=output['line'].reshape(output['line'].shape[0],6,4,*output['line'].shape[-2:])
        line=F.smooth_l1_loss(values[bi,label,:,y,x].float(),torch.stack(truth))
    else:line=output['line'].sum()*0
    losses['line']=line
    losses['total']=losses['exist_type']+losses['center']+region_weight*losses['region']+line_weight*line
    return losses
