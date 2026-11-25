"""Small, mergeable OCR adapters and deployment-identical crop preparation.

The pretrained CNN and recurrent weights remain frozen. No dictionary decoding.
"""
from collections import defaultdict
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F


class ResidualLinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        if rank < 1:
            raise ValueError('Positive adapter rank required')
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False).to(base.weight)
        self.up = nn.Linear(rank, base.out_features, bias=False).to(base.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x))

    def delta_weight(self):
        return self.up.weight @ self.down.weight

    def merged(self):
        layer = nn.Linear(self.base.in_features, self.base.out_features,
                          bias=self.base.bias is not None).to(self.base.weight)
        with torch.no_grad():
            layer.weight.copy_(self.base.weight + self.delta_weight())
            if layer.bias is not None:
                layer.bias.copy_(self.base.bias)
        return layer


class ResidualConv2d(nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        if base.groups!=1 or base.padding_mode!='zeros' or rank<1:
            raise ValueError('Unsupported OCR convolution adapter')
        self.base=base.requires_grad_(False)
        self.down=nn.Conv2d(base.in_channels,rank,base.kernel_size,stride=base.stride,
                            padding=base.padding,dilation=base.dilation,bias=False).to(base.weight)
        self.up=nn.Conv2d(rank,base.out_channels,1,bias=False).to(base.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self,x):
        return self.base(x)+self.up(self.down(x))

    def delta_weight(self):
        return torch.einsum('or,rihw->oihw',self.up.weight[:,:,0,0],self.down.weight)

    def merged(self):
        base=self.base
        layer=nn.Conv2d(base.in_channels,base.out_channels,base.kernel_size,stride=base.stride,
                        padding=base.padding,dilation=base.dilation,bias=base.bias is not None).to(base.weight)
        with torch.no_grad():
            layer.weight.copy_(base.weight+self.delta_weight())
            if layer.bias is not None:layer.bias.copy_(base.bias)
        return layer


def install_adapters(recognizer, rank=8, visual_rank=0):
    recognizer.requires_grad_(False)
    names = [n for n, m in recognizer.named_modules()
             if isinstance(m, nn.Linear) and (n.startswith('SequenceModeling.') or n == 'Prediction')]
    if not names:
        raise ValueError('No supported sequence/output projections')
    for name in names:
        parent, _, leaf = name.rpartition('.')
        module = recognizer.get_submodule(parent) if parent else recognizer
        setattr(module, leaf, ResidualLinear(getattr(module, leaf), rank))
    if visual_rank:
        conv_names=[n for n,m in recognizer.named_modules() if isinstance(m,nn.Conv2d) and n.startswith('FeatureExtraction.')]
        for name in conv_names:
            parent,_,leaf=name.rpartition('.')
            module=recognizer.get_submodule(parent)
            setattr(module,leaf,ResidualConv2d(getattr(module,leaf),visual_rank))
        names.extend(conv_names)
    return names


def merge_adapters(recognizer):
    names = [n for n, m in recognizer.named_modules() if isinstance(m, (ResidualLinear,ResidualConv2d))]
    for name in names:
        parent, _, leaf = name.rpartition('.')
        module = recognizer.get_submodule(parent) if parent else recognizer
        setattr(module, leaf, getattr(module, leaf).merged())


def merged_state(recognizer):
    state = {k: v.detach().cpu().clone() for k, v in recognizer.state_dict().items()}
    for name, module in recognizer.named_modules():
        if not isinstance(module, (ResidualLinear,ResidualConv2d)):
            continue
        for k in list(state):
            if k.startswith(name + '.'):
                del state[k]
        for k, v in module.merged().state_dict().items():
            state[name + '.' + k] = v.detach().cpu().clone()
    return state


def prepare_crop(image):
    """Exact same geometric preprocessing as PretrainedEasyOCR.recognize_crop."""
    from easyocr.utils import get_image_list
    gray = np.asarray(image.convert('L'))
    items, width = get_image_list([[0, gray.shape[1], 0, gray.shape[0]]], [], gray, model_height=64)
    return Image.fromarray(items[0][1], 'L'), int(width)


def deployment_batches(items, device):
    """Bucket by inference width: extra batch padding changes BiLSTM context."""
    from easyocr.recognition import AlignCollate
    groups = defaultdict(list)
    for image, text in items:
        crop, width = prepare_crop(image)
        groups[width].append((crop, text))
    for width, group in sorted(groups.items()):
        images = AlignCollate(imgH=64, imgW=width, keep_ratio_with_pad=True)([im for im, _ in group])
        yield images.to(device), [t for _, t in group]


def deployment_augmentation(image, rng):
    # Existing native crops have 3px margin. Add up to one pixel and the
    # deployed scale/padding. Never erode glyphs, threshold or blur the image.
    margin = rng.choice([0, 1])
    image = ImageOps.expand(image, border=margin, fill='white')
    scale = rng.choice([1, 2, 3, 4])
    return ImageOps.expand(image.resize((image.width*scale, image.height*scale),
                                       Image.Resampling.LANCZOS), border=4, fill='white')


class NativeContext:
    """Sample real neighboring bond pixels from screened TRAIN images only."""
    def __init__(self, dataset, records):
        self.labels={str((dataset/r['image']).resolve()):r['labels'] for r in records if r['split']=='train'}

    @staticmethod
    @lru_cache(maxsize=128)
    def image(path):
        with Image.open(path) as im:return im.convert('L').copy()

    def crop(self, row, rng):
        from pathlib import Path
        path=str(Path(row['source_image']).resolve())
        if row['split']!='train' or path not in self.labels:
            raise ValueError('Context crop is not a screened training image')
        im=self.image(path);x,y,u,v=row['source_box'];margin=rng.choice([2,3,4,6,8])
        # Never deliberately include another supervised text label.
        while margin>0 and any(l['atom_index']!=row['atom_index'] and
              min(u+margin,l['box'][2])>max(x-margin,l['box'][0]) and
              min(v+margin,l['box'][3])>max(y-margin,l['box'][1]) for l in self.labels[path]):
            margin-=1
        box=(max(0,int(x-margin)),max(0,int(y-margin)),min(im.width,int(np.ceil(u+margin))),min(im.height,int(np.ceil(v+margin))))
        return im.crop(box)


def teacher_consistency(student, teacher, mask, temperature=2.):
    """Distill only samples whose teacher string agrees with training truth."""
    if not mask.any():
        return student.sum()*0
    s = student[mask].float() / temperature
    t = teacher[mask].detach().float() / temperature
    return F.kl_div(s.log_softmax(-1), t.softmax(-1), reduction='none').sum(-1).mean()*temperature**2
