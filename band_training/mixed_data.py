"""Streaming Indigo mixed train/val adapter with RGB-preserving letterbox geometry."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter
from .model import targets_for

ATOM_CLASSES=['other','C','O','N','Cl','Br','S','F','B','I','P','H','Si']
CHARGE_CLASSES={-4:13,-2:14,-1:15,1:16,2:17}


def letterbox_transform(width,height,size=640):
    scale=min(size/width,size/height)
    nw,nh=round(width*scale),round(height*scale)
    return np.array([nw/width,nh/height]),np.array([(size-nw)//2,(size-nh)//2]),(nw,nh)


def prepare_annotation(annotation,band_target,size=640):
    width,height=annotation['width'],annotation['height']
    scale,offset,_=letterbox_transform(width,height,size)
    def point(p):return (np.asarray(p,float)*scale+offset).tolist()
    def box(b):
        values=np.asarray(b,float).reshape(2,2)*scale+offset
        values=np.clip(values,0,size);lo,hi=values
        if not np.all(hi>lo):raise ValueError('Degenerate scaled box')
        return [*lo.tolist(),*(hi-lo).tolist()]
    atoms=[];charges=[];ids={a['id']:i for i,a in enumerate(annotation['atoms'])}
    for a in annotation['atoms']:
        label=ATOM_CLASSES.index(a['symbol']) if a['symbol'] in ATOM_CLASSES else 0
        atoms.append(dict(index=ids[a['id']],symbol=a['symbol'],category_id=label,
                          bbox=box(a['bbox']),center=point(a['center']),formal_charge=a['formal_charge']))
    for q in annotation['charges']:
        value=int(q['value'])
        if value not in CHARGE_CLASSES:raise ValueError('Unsupported charge class: '+str(value))
        charges.append(dict(atom_index=ids[q['atom_id']],category_id=CHARGE_CLASSES[value],
                            bbox=box(q['bbox']),formal_charge=value))
    by_bond={b['bond_id']:b['category_id'] for b in annotation['bands']}
    bonds=[dict(index=b['id'],begin=ids[b['source']],end=ids[b['target']],
                category_id=by_bond[b['id']]) for b in annotation['bonds']]
    labels=list(band_target['labels'])
    quads=(np.asarray(band_target['quads'],float).reshape(-1,4,2)*scale+offset).tolist()
    assert len(labels)==len(quads)==len(bonds)
    assert all(l in range(6) for l in labels)
    boxes=torch.tensor([a['bbox'] for a in atoms]+[q['bbox'] for q in charges],dtype=torch.float32).reshape(-1,4)
    boxes[:,:2]+=boxes[:,2:]/2;boxes/=size
    detection=dict(labels=torch.tensor([a['category_id'] for a in atoms]+[q['category_id'] for q in charges]),boxes=boxes)
    return dict(atoms=atoms,charges=charges,bonds=bonds,image_size=[size,size]),detection,labels,quads


class MixedIndigoDataset(Dataset):
    def __init__(self,root,split,augment=False,limit=0,manifest=None):
        self.root=Path(root);self.split=split
        self.manifest=Path(manifest) if manifest else self.root/f'{split}.csv'
        if not self.manifest.is_file() and manifest is None:self.manifest=self.root/f'{split}.json'
        if self.manifest.suffix=='.csv':
            with self.manifest.open(newline='') as handle:self.records=list(csv.DictReader(handle))
            for row in self.records:
                for key in ['is_color','dummy_applied','has_charge']:
                    if key in row:row[key]=str(row[key]).lower() in ['true','1','yes']
        else:self.records=json.loads(self.manifest.read_text())
        if limit:self.records=self.records[:limit]
        self.augment=augment
        self.jitter=ColorJitter(brightness=.10,contrast=.10,saturation=.10,hue=.015)
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        record=self.records[index]
        annotation=json.loads((self.root/record['annotation']).read_text())
        target=json.loads((self.root/record['band_target']).read_text())
        with Image.open(self.root/record['image']) as handle:original=handle.convert('RGB')
        assert original.size==(annotation['width'],annotation['height'])
        scale,offset,shape=letterbox_transform(*original.size)
        canvas=Image.new('RGB',(640,640),'white')
        canvas.paste(original.resize(shape,Image.Resampling.LANCZOS),tuple(offset))
        if self.augment:canvas=self.jitter(canvas)
        metric,boxes,labels,quads=prepare_annotation(annotation,target)
        band=targets_for(labels,quads,'cpu',resolve_collisions=True)
        assert len(band[1])==len(labels),'Band targets must never be dropped'
        return TF.to_tensor(canvas),dict(boxes=boxes,band=band,ann=metric,labels=labels,quads=quads,record=record)


def collate(items):return torch.stack([r[0] for r in items]),[r[1] for r in items]


def worker_init(worker_id):torch.set_num_threads(1)
