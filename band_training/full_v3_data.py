"""Full-v3 paired band/line adapter; one shared RGB letterbox transform."""
import json
import numpy as np
from .mixed_data import MixedIndigoDataset,letterbox_transform
from .model import targets_for


def scaled_lines(annotation,band_target,line_target,size=640):
    if band_target['labels']!=line_target['labels']:raise ValueError('Unpaired band/line labels')
    scale,offset,_=letterbox_transform(annotation['width'],annotation['height'],size)
    lines=np.asarray(line_target['lines'],float).reshape(-1,2,2)
    if len(lines)!=len(band_target['quads']):raise ValueError('Unpaired band/line counts')
    if not np.isfinite(lines).all():raise ValueError('Nonfinite line coordinates')
    return (lines*scale+offset).tolist()


def orient_targets(labels,quads,lines,mode='canonical'):
    """Fix a visible image-coordinate convention, not a min-over-swaps loss.

    Ordinary bonds have no directed chemistry: left endpoint first, then top
    for vertical lines. Wedge/dash keep the annotation's directed line ends.
    Quad ends are aligned to the chosen line direction for ALL bond types.
    Source files and the unordered atom connectivity never change.
    """
    if mode not in ['source','canonical']:raise ValueError(mode)
    result_q=[];result_l=[]
    for label,quad,line in zip(labels,quads,lines):
        q=np.asarray(quad,float).copy();l=np.asarray(line,float).copy()
        if mode=='canonical':
            if label not in [4,5]:
                delta=l[1]-l[0]
                if delta[0]<-1e-4 or (abs(delta[0])<=1e-4 and delta[1]<0):l=l[::-1].copy()
            if np.dot(q[2:].mean(0)-q[:2].mean(0),l[1]-l[0])<0:q=np.roll(q,-2,axis=0)
        result_q.append(q.tolist());result_l.append(l.tolist())
    return result_q,result_l


class FullV3Dataset(MixedIndigoDataset):
    def __init__(self,*args,endpoint_order='canonical',**kwargs):
        super().__init__(*args,**kwargs);self.endpoint_order=endpoint_order

    def __getitem__(self,index):
        image,meta=super().__getitem__(index)
        record=meta['record']
        # Coordinate targets only. Atom IDs stay in ann for scoring and never
        # enter the bond criterion or decoder.
        annotation=json.loads((self.root/record['annotation']).read_text())
        line=json.loads((self.root/record['line_target']).read_text())
        band={'labels':meta['labels'],'quads':meta['quads']}
        meta['lines']=scaled_lines(annotation,band,line)
        meta['quads'],meta['lines']=orient_targets(meta['labels'],meta['quads'],meta['lines'],self.endpoint_order)
        meta['band']=targets_for(meta['labels'],meta['quads'],'cpu',resolve_collisions=True)
        return image,meta
