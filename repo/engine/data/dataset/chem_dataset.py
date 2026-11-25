import json
import os
from collections import defaultdict

import torch
import torch.utils.data
import torchvision.transforms.functional as TF

from PIL import Image

from ...core import register


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


@register()
class ChemDualTaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_folder,
        atom_ann_file,
        bond_ann_file,
        rgroup_ann_file=None,
        image_size=(640, 640),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        merge_rgroups_to_box_other=True,
    ):
        self.img_folder = img_folder
        self.atom_ann_file = atom_ann_file
        self.bond_ann_file = bond_ann_file
        self.rgroup_ann_file = rgroup_ann_file
        self.mean = mean
        self.std = std
        self.merge_rgroups_to_box_other = merge_rgroups_to_box_other

        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)

        self.records = self._load_records()
        self.ids = sorted(self.records.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    def set_epoch(self, epoch):
        self._epoch = epoch

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _annotations_and_image_meta(self, data):
        if isinstance(data, dict):
            images = {str(img['id']): img for img in data.get('images', [])}
            return data.get('annotations', []), images
        return data, {}

    def _load_records(self):
        atom_data = self._load_json(self.atom_ann_file)
        bond_data = self._load_json(self.bond_ann_file)
        rgroup_data = self._load_json(self.rgroup_ann_file) if self.rgroup_ann_file and os.path.exists(self.rgroup_ann_file) else []
        atom_anns, atom_images = self._annotations_and_image_meta(atom_data)
        bond_anns, bond_images = self._annotations_and_image_meta(bond_data)
        rgroup_anns, rgroup_images = self._annotations_and_image_meta(rgroup_data)

        records = defaultdict(lambda: {
            'image_id': None,
            'file_name': None,
            'width': None,
            'height': None,
            'boxes': [],
            'labels': [],
            'lines': [],
            'line_labels': [],
        })

        def ensure_meta(ann, image_meta):
            image_id = str(ann['image_id'])
            meta = image_meta.get(image_id, {})
            rec = records[image_id]
            rec['image_id'] = int(image_id) if image_id.isdigit() else image_id
            rec['file_name'] = ann.get('file_name', meta.get('file_name'))
            rec['width'] = int(ann.get('width', meta.get('width')))
            rec['height'] = int(ann.get('height', meta.get('height')))
            return rec

        for ann in atom_anns:
            rec = ensure_meta(ann, atom_images)
            rec['boxes'].append(_xywh_to_xyxy(ann['bbox']))
            rec['labels'].append(int(ann['category_id']))

        if self.merge_rgroups_to_box_other:
            for ann in rgroup_anns:
                rec = ensure_meta(ann, rgroup_images)
                rec['boxes'].append(_xywh_to_xyxy(ann['bbox']))
                rec['labels'].append(0)

        for ann in bond_anns:
            rec = ensure_meta(ann, bond_images)
            rec['lines'].append(list(ann['line']))
            rec['line_labels'].append(int(ann['category_id']))

        return records

    def __len__(self):
        return len(self.ids)

    def _scale_boxes_xyxy(self, boxes, sx, sy):
        if boxes.numel() == 0:
            return boxes
        scale = torch.tensor([sx, sy, sx, sy], dtype=torch.float32)
        return boxes * scale

    def _scale_lines_xyxy(self, lines, sx, sy):
        if lines.numel() == 0:
            return lines
        scale = torch.tensor([sx, sy, sx, sy], dtype=torch.float32)
        return lines * scale

    def _normalize_boxes_cxcywh(self, boxes_xyxy, width, height):
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy.reshape(0, 4)
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        boxes = torch.stack([
            (x1 + x2) * 0.5 / width,
            (y1 + y2) * 0.5 / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ], dim=-1)
        return boxes.clamp(0.0, 1.0)

    def _normalize_lines_xyxy(self, lines_xyxy, width, height):
        if lines_xyxy.numel() == 0:
            return lines_xyxy.reshape(0, 4)
        norm = torch.tensor([width, height, width, height], dtype=torch.float32)
        return (lines_xyxy / norm).clamp(0.0, 1.0)

    def _build_target(self, rec):
        out_w, out_h = self.image_size
        sx = out_w / rec['width']
        sy = out_h / rec['height']

        boxes_xyxy = torch.tensor(rec['boxes'], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(rec['labels'], dtype=torch.int64)
        lines_xyxy = torch.tensor(rec['lines'], dtype=torch.float32).reshape(-1, 4)
        line_labels = torch.tensor(rec['line_labels'], dtype=torch.int64)

        boxes_xyxy = self._scale_boxes_xyxy(boxes_xyxy, sx, sy)
        lines_xyxy = self._scale_lines_xyxy(lines_xyxy, sx, sy)

        target = {
            'boxes': self._normalize_boxes_cxcywh(boxes_xyxy, out_w, out_h),
            'labels': labels,
            'lines': self._normalize_lines_xyxy(lines_xyxy, out_w, out_h),
            'line_labels': line_labels,
            'image_id': torch.tensor([rec['image_id']]),
            'orig_size': torch.tensor([out_w, out_h]),
            'size': torch.tensor([out_w, out_h]),
            'area': ((boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])) if boxes_xyxy.numel() > 0 else torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((labels.shape[0],), dtype=torch.int64),
        }
        return target, boxes_xyxy, lines_xyxy

    def get_ground_truth(self, image_id):
        rec = self.records[str(int(image_id) if not isinstance(image_id, str) or str(image_id).isdigit() else image_id)]
        target, boxes_xyxy, lines_xyxy = self._build_target(rec)
        return {
            'boxes': boxes_xyxy,
            'labels': target['labels'],
            'lines': lines_xyxy,
            'line_labels': target['line_labels'],
            'size': target['orig_size'],
        }

    def __getitem__(self, idx):
        record_id = self.ids[idx]
        rec = self.records[record_id]
        img_path = os.path.join(self.img_folder, rec['file_name'])

        image = Image.open(img_path).convert('RGB')
        image = TF.resize(image, [self.image_size[1], self.image_size[0]])
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        target, _, _ = self._build_target(rec)
        target['idx'] = torch.tensor([idx])

        return image, target
