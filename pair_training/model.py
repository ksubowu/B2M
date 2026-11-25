"""Frozen existing model + trainable image-conditioned atom-pair head."""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_convert, nms

PAIR_CLASSES = ['none', 'other', 'single', 'double', 'triple', 'aromatic',
                'wedge_forward', 'dash_forward', 'wedge_reverse', 'dash_reverse']


def knn_pairs(centers, k=6):
    """Undirected union of directed KNN; never assume proximity means a bond."""
    n = len(centers)
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long, device=centers.device)
    distances = torch.cdist(centers, centers)
    distances.fill_diagonal_(float('inf'))
    nearest = distances.argsort(dim=1, stable=True)[:, :min(k, n - 1)]
    source = torch.arange(n, device=centers.device)[:, None].expand_as(nearest)
    return torch.unique(torch.stack((source.flatten(), nearest.flatten()), -1).sort(-1).values, dim=0)


def encode_bond(category, reverse=False):
    return category + 1 + (2 if reverse and category in (5, 6) else 0)


def decode_bond(label):
    if label == 0:
        return None
    return (label - 3, True) if label in (8, 9) else (label - 1, False)


@torch.no_grad()
def predicted_atoms(outputs, index, cutoff=.4, nms_iou=.5, max_atoms=300):
    # Atom categories 0..12; charge detections 13..17 are not extra atoms.
    scores, labels = outputs['pred_logits'][index, :, :13].sigmoid().max(-1)
    boxes = outputs['pred_boxes'][index]
    keep = torch.where(scores >= cutoff)[0]
    keep = keep[nms(box_convert(boxes[keep], 'cxcywh', 'xyxy'), scores[keep], nms_iou)[:max_atoms]]
    return {'centers': boxes[keep, :2].detach(), 'labels': labels[keep].detach(),
            'scores': scores[keep].detach(), 'boxes': boxes[keep].detach()}


class AtomPairHead(nn.Module):
    def __init__(self, channels=256, projection=32, hidden=256, samples=11):
        super().__init__()
        self.samples = samples
        self.projections = nn.ModuleList([nn.Conv2d(channels, projection, 1) for _ in range(3)])
        self.atom_embedding = nn.Embedding(13, 16)
        # Three parallel sampling paths plus two endpoint samples, on 3 scales.
        inputs = projection * 3 * (samples * 3 + 2) + 32 + 5
        self.classifier = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(),
                                        nn.Dropout(.1), nn.Linear(hidden, len(PAIR_CLASSES)))

    def forward(self, features, atoms, pairs):
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        results = []
        for batch_index, (atom, pair) in enumerate(zip(atoms, pairs)):
            if len(pair) == 0:
                # Keep a differentiable zero for rare empty-pair training batches.
                results.append(projected[0][batch_index].sum().expand(0, len(PAIR_CLASSES)))
                continue
            centers = atom['centers']
            a, b = centers[pair[:, 0]], centers[pair[:, 1]]
            delta = b - a
            length = delta.norm(dim=-1, keepdim=True).clamp_min(1e-5)
            normal = torch.stack((-delta[:, 1], delta[:, 0]), -1) / length
            t = torch.linspace(0, 1, self.samples, device=a.device)
            path = a[:, None, :] + delta[:, None, :] * t[None, :, None]
            width = .005 + .02 * length
            offsets = torch.tensor([-1., 0., 1.], device=a.device)
            strip = path[:, :, None, :] + (normal * width)[:, None, None, :] * offsets[None, None, :, None]
            grid = torch.cat((strip.flatten(1, 2), a[:, None, :], b[:, None, :]), dim=1)
            grid = grid.mul(2).sub(1)[None]
            sampled = [F.grid_sample(feature[batch_index:batch_index+1], grid,
                                     align_corners=False, padding_mode='border')[0].permute(1, 2, 0).flatten(1)
                       for feature in projected]
            labels = atom['labels'].clamp(0, 12)
            embedding = self.atom_embedding(labels[pair]).flatten(1)
            geometry = torch.cat((delta, length, delta / length), dim=-1)
            results.append(self.classifier(torch.cat((*sampled, embedding, geometry), dim=-1)))
        return results


class PixelPairRefiner(nn.Module):
    """Independent existence/type refinement from oriented high-resolution ink.

    The frozen old head is a prior, not a hard proposal gate. Every KNN pair is
    checked, including pairs whose old head calls 'none'.
    """
    requires_images = True

    def __init__(self, prior_temperature=4., patch_height=48, patch_width=96):
        super().__init__()
        self.old_head = AtomPairHead().requires_grad_(False).eval()
        self.prior_temperature = prior_temperature
        self.patch_height, self.patch_width = patch_height, patch_width
        self.pixel_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 8)), nn.Flatten(), nn.Linear(96*3*8, 128), nn.GELU())
        self.geometry = nn.Sequential(nn.Linear(5, 32), nn.GELU())
        self.type_delta = nn.Linear(160, 9)
        self.existence_delta = nn.Linear(160, 1)
        for layer in [self.type_delta, self.existence_delta]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def train(self, mode=True):
        super().train(mode)
        self.old_head.eval()
        return self

    def oriented_patches(self, image, centers, pairs):
        a, b = centers[pairs[:, 0]], centers[pairs[:, 1]]
        delta = b-a
        length = delta.norm(dim=-1, keepdim=True).clamp_min(1e-5)
        normal = torch.stack((-delta[:, 1], delta[:, 0]), -1) / length
        longitudinal = torch.linspace(-.2, 1.2, self.patch_width, device=image.device)
        transverse = torch.linspace(-.5, .5, self.patch_height, device=image.device)
        # A wide crop sees inner ring strokes, wedge width, and individual dash
        # marks. A minimum width protects short/near-coincident atom pairs.
        crop_width = (.9 * length).clamp_min(.018)
        grid = (a[:, None, None, :] + delta[:, None, None, :] * longitudinal[None, None, :, None]
                + (normal * crop_width)[:, None, None, :] * transverse[None, :, None, None])
        mean = image.new_tensor([.485, .456, .406])[:, None, None]
        std = image.new_tensor([.229, .224, .225])[:, None, None]
        rgb = (image * std + mean).clamp(0, 1)
        ink = 1-rgb.amin(0, keepdim=True)
        n = len(pairs)
        patches = F.grid_sample(ink[None].expand(n, -1, -1, -1), grid*2-1,
                                align_corners=False, padding_mode='zeros')
        geometry = torch.cat((delta, length, delta / length), -1)
        return patches, geometry

    def forward(self, features, atoms, pairs, images):
        with torch.no_grad():
            priors = self.old_head(features, atoms, pairs)
        results = []
        for image, atom, pair, prior in zip(images, atoms, pairs, priors):
            if not len(pair):
                results.append(prior + self.type_delta.weight.sum()*0)
                continue
            patches, geometry = self.oriented_patches(image, atom['centers'], pair)
            embedding = torch.cat((self.pixel_encoder(patches), self.geometry(geometry)), -1)
            existence_prior = (torch.logsumexp(prior[:, 1:], -1) - prior[:, 0]).clamp(-8, 8)
            existence = existence_prior + self.existence_delta(embedding).squeeze(-1)
            types = prior[:, 1:] / self.prior_temperature + self.type_delta(embedding)
            # Normalized logits maintain the existing checkpoint/prediction
            # interface; separate task losses prevent class ambiguity deleting a
            # connection that is confidently present.
            results.append(torch.cat((F.logsigmoid(-existence)[:, None],
                                      F.logsigmoid(existence)[:, None] + F.log_softmax(types, -1)), -1))
        return results


def run_pair_head(head, features, atoms, pairs, images):
    if getattr(head, 'requires_images', False):
        return head(features, atoms, pairs, images=images)
    return head(features, atoms, pairs)


def pair_decision_policy(head, atom, checkpoint):
    """Resolve threshold/mode for adaptive and standard pair-head checkpoints."""
    if getattr(head, 'is_resolution_adaptive', False):
        coarse = head.coarse(atom)
        if coarse:
            return float(checkpoint['selected_validation_threshold']), 'existence', 'coarse'
        return float(head.clear_threshold), 'max_type', 'clear'
    return (float(checkpoint['selected_validation_threshold']),
            checkpoint.get('args', {}).get('score_mode', 'max_type'), 'single_head')


class ResolutionAdaptivePairHead(nn.Module):
    """Two validated raster-domain specialists, sharing the frozen image encoder.

    Coarse and clear routing uses input raster resolution, never GT or SMILES.
    Clear images keep both the old head and its original decision threshold.
    """
    is_resolution_adaptive = True

    def __init__(self):
        super().__init__()
        self.clear_head = AtomPairHead()
        self.coarse_head = AtomPairHead()
        # 320 is the smallest native raster in the original training renders;
        # coarse augmentation covers 160..288. Keep the boundary explicit.
        self.register_buffer('clear_resolution_min', torch.tensor(320.))
        self.register_buffer('clear_threshold', torch.tensor(.7))

    def coarse(self, atom):
        return float(atom.get('raster_resolution', 640)) < float(self.clear_resolution_min)

    def forward(self, features, atoms, pairs):
        results = []
        for i, (atom, pair) in enumerate(zip(atoms, pairs)):
            head = self.coarse_head if self.coarse(atom) else self.clear_head
            results.append(head([f[i:i+1] for f in features], [atom], [pair])[0])
        return results


class FrozenEncoderPairModel(nn.Module):
    def __init__(self, base, head=None):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.pair_head = head or AtomPairHead()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Also freeze BatchNorm statistics, dropout, and denoising.
        if hasattr(self,'original_box_decoder'):
            self.original_box_decoder.eval();self.original_anchor.eval()
        return self

    @torch.no_grad()
    def extract(self, images):
        features = self.base.encoder(self.base.backbone(images))
        if isinstance(features, tuple):
            features = features[0]
        return features, self.base.box_decoder(features)

    def forward(self, images, atoms=None, k=6, atom_cutoff=.4, native_sizes=None):
        features, atom_outputs = self.extract(images)
        if atoms is None:
            atoms = [predicted_atoms(atom_outputs, i, atom_cutoff) for i in range(len(images))]
        if native_sizes is not None:
            for atom, size in zip(atoms, native_sizes):
                atom['raster_resolution'] = min(size)
        pairs = [knn_pairs(atom['centers'], k) for atom in atoms]
        logits = run_pair_head(self.pair_head, features, atoms, pairs, images)
        return logits, atoms, pairs


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
