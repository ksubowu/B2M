import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .model import encode_bond


class PairDataset(Dataset):
    def __init__(self, root, split, limit=0, allow_point_truth=False,
                 lowres_probability=0., validation_resolution=0):
        self.root = Path(root).resolve()
        self.allow_point_truth = allow_point_truth
        self.lowres_probability = lowres_probability
        self.validation_resolution = validation_resolution
        self.records = [r for r in json.loads((self.root / 'manifest.json').read_text()) if r.get('split', 'val') == split]
        if limit:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f'Empty {split} dataset')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        meta = self.records[index]
        truth = json.loads((self.root / meta['truth']).read_text())
        audited = truth.get('coordinate_audit', {}).get('status') in ('consistent', 'insufficient_layout_anchors')
        point_truth = (self.allow_point_truth and truth.get('loader') == 'rdkit_drawing' and
                       truth.get('coordinate_audit', {}).get('status') == 'not_highlight_renderer' and
                       truth.get('valid_graph_truth'))
        if not audited and not point_truth:
            raise ValueError('Requires verified pixel labels: ' + meta['file_name'])
        with Image.open(self.root / meta.get('image', f"images/val/{meta['file_name']}")) as image:
            image = image.convert('RGB')
            resolution = self.validation_resolution
            if resolution == -1:
                resolution = 0 if index % 2 == 0 else 256
            if self.lowres_probability and float(torch.rand(())) < self.lowres_probability:
                resolution = [160, 192, 224, 256, 288, 320][int(torch.randint(6, ()))]
            if resolution:
                image = image.resize((resolution, resolution), Image.Resampling.BOX).convert('L')
                # Matches the low-resolution black/white rasterization family.
                # Geometry and normalized labels are unchanged.
                image = image.point(lambda value: 255 if value > 245 else 0).convert('RGB')
            image = TF.to_tensor(TF.resize(image, [640, 640]))
        image = TF.normalize(image, [.485, .456, .406], [.229, .224, .225])
        scale = np.asarray(truth['image_size'], dtype=np.float32)
        centers = np.asarray([a['center'] for a in truth['atoms']], dtype=np.float32) / scale
        ids = {a['index']: i for i, a in enumerate(truth['atoms'])}
        edges = {(ids[b['begin']], ids[b['end']]): b['category_id'] for b in truth['bonds']}
        lengths = [np.linalg.norm(centers[i] - centers[j]) for i, j in edges]
        # Rendering uses square canvases; match tolerance follows benchmark policy.
        tolerance = max(4 / scale[0], .25 * float(np.median(lengths))) if lengths else 4 / scale[0]
        target = {'centers': torch.tensor(centers),
                  'labels': torch.tensor([a['category_id'] for a in truth['atoms']]),
                  'edges': edges, 'tolerance': tolerance, 'name': meta['file_name'],
                  'group': meta.get('group'), 'image_size': truth['image_size']}
        target['raster_augmented'] = bool(resolution)
        target['raster_resolution'] = resolution or min(truth['image_size'])
        return image, target


def collate(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def move_target(target, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}


def gt_atoms(target, jitter=0., shuffle=False):
    centers = target['centers']
    if jitter:
        centers = (centers + torch.randn_like(centers) * jitter).clamp(0, 1)
    order = torch.randperm(len(centers), device=centers.device) if shuffle else torch.arange(len(centers), device=centers.device)
    return {'centers': centers[order], 'labels': target['labels'][order], 'gt_indices': order}


def atom_mapping(atoms, target):
    if 'gt_indices' in atoms:
        return {i: int(j) for i, j in enumerate(atoms['gt_indices'])}
    if not len(atoms['centers']):
        return {}
    distance = torch.cdist(atoms['centers'], target['centers']).detach().cpu().numpy()
    rows, cols = linear_sum_assignment(np.where(distance <= target['tolerance'], distance, 1e6))
    return {int(i): int(j) for i, j in zip(rows, cols) if distance[i, j] <= target['tolerance']}


def pair_targets(atoms, pairs, target, ignore_unmatched=True):
    mapping = atom_mapping(atoms, target)
    values = []
    for i, j in pairs.detach().cpu().tolist():
        a, b = mapping.get(i), mapping.get(j)
        if a is None or b is None:
            values.append(-100 if ignore_unmatched else 0)
        elif (a, b) in target['edges']:
            values.append(encode_bond(target['edges'][a, b]))
        elif (b, a) in target['edges']:
            values.append(encode_bond(target['edges'][b, a], reverse=True))
        else:
            values.append(0)
    return torch.tensor(values, dtype=torch.long, device=pairs.device), mapping


def verify_split(root):
    records = json.loads((Path(root) / 'manifest.json').read_text())
    groups = {split: {r['group'] for r in records if r['split'] == split} for split in ['train', 'val']}
    if groups['train'] & groups['val']:
        raise ValueError('Molecular connectivity overlaps train and validation')
    return {split: len(group) for split, group in groups.items()}
