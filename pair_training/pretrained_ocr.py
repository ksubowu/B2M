"""Frozen EasyOCR English recognizer as an optional PyTorch submodule.

Reuse its entire pretrained visual encoder + BiLSTM + CTC classifier. No new
training is performed by this wrapper; an adapted checkpoint may be loaded. Whole-label
boxes must come from the caller; this adapter does not detect/group text.
"""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageOps

from .text_sequence import ALPHANUMERIC_CHARACTERS, TextVocabulary


class PretrainedEasyOCR(nn.Module):
    def __init__(self, model_dir=None, device='cpu', download_enabled=False, reader=None, checkpoint=None):
        super().__init__()
        if reader is None:
            import easyocr
            reader=easyocr.Reader(['en'],gpu=False,detector=False,quantize=False,verbose=False,
                model_storage_directory=str(Path(model_dir).resolve()) if model_dir else None,
                download_enabled=download_enabled)
        self.recognizer=reader.recognizer.requires_grad_(False).eval()
        self.vocabulary=TextVocabulary(reader.character)
        self.backend='pretrained_easyocr_english_g2'
        if checkpoint is not None:
            adapted=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if adapted['characters']!=reader.character:
                raise ValueError('Adapted OCR character order differs from pretrained model')
            self.recognizer.load_state_dict(adapted['recognizer'],strict=True)
            self.backend='adapted_easyocr_chemical_v1'
        supported=set(reader.character)
        if not set(ALPHANUMERIC_CHARACTERS)<=supported:
            raise ValueError('Pretrained recognizer does not cover all 62 characters')
        allowed=[True]+[c in ALPHANUMERIC_CHARACTERS for c in reader.character]
        # Preserve pretrained output classes/order; masking is inference-only.
        self.register_buffer('allowed',torch.tensor(allowed,dtype=torch.bool))
        self.to(device)

    def train(self,mode=True):
        super().train(False)
        return self

    def forward(self,crops):
        """N,1,64,W grayscale crops normalized to [-1,1] -> pretrained logits."""
        if crops.ndim!=4 or crops.shape[1]!=1 or crops.shape[2]!=64:
            raise ValueError('Expected N,1,64,W preprocessed crops')
        text=torch.zeros((len(crops),26),dtype=torch.long,device=crops.device)
        logits=self.recognizer(crops,text)
        return logits.masked_fill(~self.allowed[None,None,:],float('-inf'))

    def decode(self,logits):
        """Match EasyOCR greedy CTC text and its path-confidence formula."""
        results=self.vocabulary.decode(logits)
        scores,ids=logits.detach().float().softmax(-1).max(-1)
        for result,values,indices in zip(results,scores,ids):
            selected=values[indices!=0]
            result['score']=float((selected.clamp_min(1e-30).log().sum()*2/math.sqrt(len(selected))).exp()) if len(selected) else 0.
            result['backend']=self.backend
        return results

    @torch.no_grad()
    def recognize_crop(self,rgb):
        """Callback for notebook_ocr.recognize_multiscale; crop is already padded."""
        from easyocr.utils import get_image_list
        from easyocr.recognition import AlignCollate
        import cv2
        array=np.asarray(rgb)
        gray=cv2.cvtColor(array,cv2.COLOR_RGB2GRAY) if array.ndim==3 else array
        items,width=get_image_list([[0,gray.shape[1],0,gray.shape[0]]],[],gray,model_height=64)
        batch=AlignCollate(imgH=64,imgW=int(width),keep_ratio_with_pad=True)([Image.fromarray(items[0][1],'L')])
        result=self.decode(self(batch.to(self.allowed.device)))[0]
        return result['raw_text'],result['score']

    @torch.no_grad()
    def recognize_regions(self,image,boxes,scales=(1,2,3,4),margin=3):
        """Native-pixel xyxy boxes -> literal strings; no alias dictionary.

        Uses EasyOCR's own crop preprocessing. Scale selection compares only
        this backend's scores. Raw scores are not comparable to PaddleOCR.
        """
        self.eval();image=image.convert('RGB');outputs=[]
        for box in boxes:
            x1,y1,x2,y2=map(float,box)
            if not np.isfinite([x1,y1,x2,y2]).all() or x2<=x1 or y2<=y1:
                raise ValueError('Invalid text region')
            crop_box=[max(0,math.floor(x1-margin)),max(0,math.floor(y1-margin)),
                      min(image.width,math.ceil(x2+margin)),min(image.height,math.ceil(y2+margin))]
            if crop_box[2]<=crop_box[0] or crop_box[3]<=crop_box[1]:
                raise ValueError('Text region outside image')
            crop=image.crop(crop_box);candidates=[]
            for scale in scales:
                if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid scale')
                enlarged=ImageOps.expand(crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),
                                                     Image.Resampling.LANCZOS),border=4,fill='white')
                text,score=self.recognize_crop(np.asarray(enlarged))
                result=dict(text=text,raw_text=text,score=score,backend=self.backend)
                candidates.append(dict(**result,scale=scale))
            if not candidates:raise ValueError('At least one scale is required')
            best=max(candidates,key=lambda r:r['score'] if r['text'] and math.isfinite(r['score']) else -1)
            outputs.append(dict(**best,box=list(box),crop_box=crop_box,candidates=candidates,
                                selection='highest_score_within_easyocr',pretrained=True,word_lexicon=False))
        return outputs


def labels_from_regions(predictions,atom_indices):
    if len(predictions)!=len(atom_indices) or len(set(atom_indices))!=len(atom_indices):
        raise ValueError('One distinct graph node per region')
    return {int(index):prediction for index,prediction in zip(atom_indices,predictions)}
