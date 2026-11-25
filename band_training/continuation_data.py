"""Identical v3/v6 transforms with only the targets used by scored training."""
import json
from PIL import Image
from torchvision.transforms import functional as TF
from .mixed_data import MixedIndigoDataset,prepare_annotation,letterbox_transform
from .full_v3_data import scaled_lines,orient_targets
from .scored_bond_model import scored_targets


class ContinuationDataset(MixedIndigoDataset):
    def __init__(self,*args,endpoint_order='canonical',**kwargs):
        super().__init__(*args,**kwargs);self.endpoint_order=endpoint_order

    def __getitem__(self,index):
        record=self.records[index];raw=json.loads((self.root/record['annotation']).read_text())
        band=json.loads((self.root/record['band_target']).read_text());line=json.loads((self.root/record['line_target']).read_text())
        with Image.open(self.root/record['image']) as im:original=im.convert('RGB')
        assert original.size==(raw['width'],raw['height'])
        scale,offset,shape=letterbox_transform(*original.size);canvas=Image.new('RGB',(640,640),'white')
        canvas.paste(original.resize(shape,Image.Resampling.LANCZOS),tuple(offset))
        if self.augment:canvas=self.jitter(canvas)
        ann,boxes,labels,quads=prepare_annotation(raw,band);lines=scaled_lines(raw,band,line)
        quads,lines=orient_targets(labels,quads,lines,self.endpoint_order)
        return TF.to_tensor(canvas),dict(ann=ann,boxes=boxes,labels=labels,quads=quads,lines=lines,record=record,
            scored_bonds=scored_targets(labels,quads,lines))
