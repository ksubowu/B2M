"""Internal, case-sensitive CTC recognition from the shared encoder pyramid.

Boxes must cover whole labels. This module neither detects boxes nor invokes an
external OCR engine. Coordinates are normalized xyxy in the encoder input frame.
"""
import string

import torch
from torch import nn
from torch.nn import functional as F


ALPHANUMERIC_CHARACTERS = string.ascii_lowercase + string.ascii_uppercase + string.digits
# New models predict 62 literal characters plus CTC blank, not alias classes.
# Existing checkpoints retain their explicitly stored config.characters.
DEFAULT_CHARACTERS = ALPHANUMERIC_CHARACTERS


def alphabet_coverage(texts, characters=DEFAULT_CHARACTERS):
    from collections import Counter
    counts = Counter(''.join(texts))
    return dict(characters=characters, counts={c: counts[c] for c in characters},
                missing=''.join(c for c in characters if not counts[c]),
                unsupported=''.join(sorted(set(counts)-set(characters))))


@torch.no_grad()
def decode_ctc_lexicon(logits, vocabulary, words):
    """Forced closed-word decoding; scores are conditional on this lexicon.

    Never use this confidence as evidence that a crop is a known word. Unseen
    strings are necessarily changed into a lexicon member by this function.
    """
    if not words or len(set(words))!=len(words):
        raise ValueError('Require a nonempty, unique training lexicon')
    targets,lengths=vocabulary.encode(words,logits.device)
    inputs=torch.full((len(words),),logits.shape[1],dtype=torch.long,device=logits.device)
    results=[]
    for row in logits:
        lp=row.float().log_softmax(-1)[:,None,:].expand(-1,len(words),-1).contiguous()
        nll=F.ctc_loss(lp,targets,inputs,lengths,reduction='none',zero_infinity=False)
        if not torch.isfinite(nll).any():raise ValueError('No CTC-alignable word')
        best=int(nll.argmin());word=words[best]
        results.append(dict(text=word,raw_text=word,score=float((-nll).softmax(0)[best]),
                            forced_closed_vocabulary=True,score_kind='conditional_lexicon_posterior'))
    return results


class TextVocabulary:
    def __init__(self, characters=DEFAULT_CHARACTERS):
        if len(set(characters)) != len(characters):
            raise ValueError('Duplicate characters')
        self.characters = characters
        self.ids = {c: i + 1 for i, c in enumerate(characters)}

    def encode(self, texts, device=None):
        # Unknown characters are errors, never silently removed.
        return (torch.tensor([self.ids[c] for s in texts for c in s],
                             dtype=torch.long, device=device),
                torch.tensor([len(s) for s in texts], dtype=torch.long, device=device))

    def decode(self, logits):
        """Greedy CTC collapse; score is an uncalibrated path confidence."""
        probabilities = logits.detach().float().softmax(-1)
        scores, indices = probabilities.max(-1)
        results = []
        for sequence, confidence in zip(indices.cpu().tolist(), scores.cpu().tolist()):
            previous, chars, kept = 0, [], []
            for index, score in zip(sequence, confidence):
                if index and index != previous:
                    chars.append(self.characters[index - 1])
                    kept.append(score)
                previous = index
            results.append({'text': ''.join(chars), 'raw_text': ''.join(chars),
                            'score': float(torch.tensor(kept).log().mean().exp()) if kept else 0.,
                            'backend': 'shared_encoder_ctc'})
        return results


class SharedFeatureTextHead(nn.Module):
    def __init__(self, characters=DEFAULT_CHARACTERS, channels=256,
                 projection=32, hidden=128, steps=48, roi_height=5):
        super().__init__()
        self.config = dict(characters=characters, channels=channels,
                           projection=projection, hidden=hidden, steps=steps,
                           roi_height=roi_height)
        self.vocabulary = TextVocabulary(characters)
        self.steps, self.roi_height = steps, roi_height
        self.projections = nn.ModuleList(nn.Conv2d(channels, projection, 1) for _ in range(3))
        self.visual = nn.Sequential(nn.Conv2d(projection * 3, hidden, 3, padding=1),
                                    nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.sequence = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden, len(characters) + 1)

    def forward(self, features, boxes):
        if len(features) != 3 or len(boxes) != features[0].shape[0]:
            raise ValueError('Expected three feature scales and boxes for each image')
        projected = [layer(f) for layer, f in zip(self.projections, features)]
        crops = []
        for batch, regions in enumerate(boxes):
            if not len(regions):
                continue
            if regions.ndim != 2 or regions.shape[1] != 4:
                raise ValueError('Boxes must have shape N,4')
            if not torch.isfinite(regions).all() or (regions[:, 2:] <= regions[:, :2]).any():
                raise ValueError('Invalid text boxes')
            if (regions < 0).any() or (regions > 1).any():
                raise ValueError('Boxes must be normalized to [0,1]')
            x = (torch.arange(self.steps, device=regions.device, dtype=regions.dtype) + .5) / self.steps
            y = (torch.arange(self.roi_height, device=regions.device, dtype=regions.dtype) + .5) / self.roi_height
            xx = regions[:, 0, None] + (regions[:, 2] - regions[:, 0])[:, None] * x
            yy = regions[:, 1, None] + (regions[:, 3] - regions[:, 1])[:, None] * y
            grid = torch.stack((xx[:, None, :].expand(-1, self.roi_height, -1),
                                yy[:, :, None].expand(-1, -1, self.steps)), -1) * 2 - 1
            crops.append(torch.cat([F.grid_sample(f[batch:batch+1].expand(len(regions), -1, -1, -1),
                                                   grid, align_corners=False)
                                    for f in projected], 1))
        if not crops:
            return projected[0].sum().expand(0, self.steps, self.classifier.out_features)
        visual = self.visual(torch.cat(crops)).mean(2).transpose(1, 2)
        sequence, _ = self.sequence(visual)
        return self.classifier(sequence)

    def loss(self, logits, texts):
        if logits.shape[0] != len(texts):
            raise ValueError('One target string per box is required')
        if not texts:
            return logits.sum()
        for text in texts:
            needed = len(text) + sum(a == b for a, b in zip(text, text[1:]))
            if needed > logits.shape[1]:
                raise ValueError('Label too long for CTC time axis: ' + text)
        labels, lengths = self.vocabulary.encode(texts, logits.device)
        inputs = torch.full((len(texts),), logits.shape[1], dtype=torch.long, device=logits.device)
        return F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), labels, inputs, lengths,
                          blank=0, zero_infinity=False)


class NativeROITextHead(SharedFeatureTextHead):
    """Joint recognizer: native-resolution pixel CNN/GRU + shared feature logits.

    Small characters need not survive the frozen detector's stride-8 feature
    maps. Native RGB crops retain their original raster detail. No external OCR.
    """
    requires_images = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden = self.config['hidden']
        self.pixel_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.pixel_sequence = nn.GRU(hidden, hidden//2, batch_first=True, bidirectional=True)
        self.pixel_classifier = nn.Linear(hidden, len(self.vocabulary.characters)+1)
        self.context_weight = nn.Parameter(torch.tensor(-2.))
        self.anchor_predictor = nn.Sequential(nn.Linear(hidden*2*12, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, features, boxes, images=None, return_anchors=False):
        if images is None or len(images) != len(boxes):
            raise ValueError('Native RGB tensors in [0,1] are required, one per image')
        context = super().forward(features, boxes)
        if not context.shape[0]:
            return (context,context.new_empty((0,2))) if return_anchors else context
        crops=[];anchor_crops=[]
        for regions, image in zip(boxes, images):
            if not len(regions):continue
            if image.ndim!=3 or image.shape[0]!=3:
                raise ValueError('Expected native CHW RGB image')
            x=(torch.arange(self.steps*4, device=regions.device, dtype=regions.dtype)+.5)/(self.steps*4)
            y=(torch.arange(32, device=regions.device, dtype=regions.dtype)+.5)/32
            xx=regions[:,0,None]+(regions[:,2]-regions[:,0])[:,None]*x
            yy=regions[:,1,None]+(regions[:,3]-regions[:,1])[:,None]*y
            grid=torch.stack((xx[:,None,:].expand(-1,32,-1),yy[:,:,None].expand(-1,-1,self.steps*4)),-1)*2-1
            # Padding outside the image is white, not a false black frame.
            gray=image.amin(0,keepdim=True).unsqueeze(0)
            crop=F.grid_sample((gray-1).expand(len(regions),-1,-1,-1),grid,align_corners=False)+1
            crops.append(crop*2-1)
            if return_anchors:
                # Include surrounding bond strokes; text alone cannot identify
                # the connected side when the same string has two orientations.
                h=regions[:,3]-regions[:,1]
                padx=h*image.shape[1]/image.shape[2]*.8
                expanded=regions+torch.stack((-padx,-h*.5,padx,h*.5),-1)
                xx=expanded[:,0,None]+(expanded[:,2]-expanded[:,0])[:,None]*x
                yy=expanded[:,1,None]+(expanded[:,3]-expanded[:,1])[:,None]*y
                extra_grid=torch.stack((xx[:,None,:].expand(-1,32,-1),yy[:,:,None].expand(-1,-1,self.steps*4)),-1)*2-1
                extra=F.grid_sample((gray-1).expand(len(regions),-1,-1,-1),extra_grid,align_corners=False)+1
                anchor_crops.append(extra*2-1)
        visual=self.pixel_encoder(torch.cat(crops)).mean(2).transpose(1,2)
        sequence,_=self.pixel_sequence(visual)
        logits=self.pixel_classifier(sequence)+self.context_weight.sigmoid()*context
        if return_anchors:
            spatial=self.pixel_encoder(torch.cat(anchor_crops))
            offsets=self.anchor_predictor(F.adaptive_avg_pool2d(spatial,(2,12)).flatten(1))
            regions=torch.cat(boxes)
            anchors=regions[:,:2]+offsets*(regions[:,2:]-regions[:,:2])
            return logits,anchors
        return logits


@torch.no_grad()
def recognize_shared_labels(head, features, boxes, atom_indices, images=None):
    """Return literal label maps suitable for notebook_ocr.labeled_cxsmiles.

    One box/index list per image; node selection and graph construction are the
    caller's responsibility. No dictionary expansion or spelling substitution.
    """
    if len(boxes) != len(atom_indices):
        raise ValueError('One index list per image')
    for regions, indices in zip(boxes, atom_indices):
        if len(regions) != len(indices) or len(set(indices)) != len(indices):
            raise ValueError('Require one distinct graph node index per box')
    was_training = head.training
    head.eval()
    try:
        kwargs = dict(images=images) if getattr(head, 'requires_images', False) else {}
        decoded = iter(head.vocabulary.decode(head(features, boxes, **kwargs)))
        return [{int(index): next(decoded) for index in indices} for indices in atom_indices]
    finally:
        head.train(was_training)
