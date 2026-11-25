from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from scipy.optimize import linear_sum_assignment

from ..core import register
from .chem_model import ChemRTv4
from .chem_postprocessor import ChemPostProcessor
from .chem_criterion import _swap_line_endpoints
from .rtdetrv2_decoder import RTDETRTransformerv2


def _line_endpoint_cost(pred_lines: torch.Tensor, tgt_lines: torch.Tensor) -> torch.Tensor:
    pred_a = pred_lines[:, :2]
    pred_b = pred_lines[:, 2:4]
    tgt_a = tgt_lines[:, :2]
    tgt_b = tgt_lines[:, 2:4]

    direct = torch.cdist(pred_a, tgt_a, p=2) + torch.cdist(pred_b, tgt_b, p=2)
    reverse = torch.cdist(pred_a, tgt_b, p=2) + torch.cdist(pred_b, tgt_a, p=2)
    return torch.minimum(direct, reverse)


def _target_box_centers(target: Dict[str, torch.Tensor]) -> torch.Tensor:
    boxes = target.get('boxes')
    if boxes is None or boxes.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=target['lines'].device)
    # Target boxes are normalized cxcywh during training.
    return boxes[:, :2]


def _assign_line_to_atom_centers(lines: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    if lines.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32, device=lines.device)
    if centers.shape[0] == 0:
        return torch.zeros((lines.shape[0], 4), dtype=torch.float32, device=lines.device)
    if centers.shape[0] == 1:
        only = centers[0].repeat(lines.shape[0], 2)
        return only

    assignments = []
    for line in lines:
        endpoints = torch.stack([line[:2], line[2:4]], dim=0)
        dist = torch.cdist(endpoints, centers, p=2)
        first = int(dist[0].argmin().item())
        second_order = torch.argsort(dist[1])
        second = int(second_order[0].item())
        if second == first:
            second = int(second_order[1].item())

        direct = dist[0, first] + dist[1, second]
        reverse = dist[0, second] + dist[1, first]
        if reverse < direct:
            pair = torch.cat([centers[second], centers[first]], dim=0)
        else:
            pair = torch.cat([centers[first], centers[second]], dim=0)
        assignments.append(pair)

    return torch.stack(assignments, dim=0)


@register()
class ChemRTv4Endpoint(ChemRTv4):
    pass


@register()
class EndpointLineTransformer(RTDETRTransformerv2):
    def __init__(
        self,
        num_endpoint_queries=None,
        line_representation='endpoints',
        endpoint_heatmap_stride=4,
        use_deformable_sampling=True,
        predict_endpoint_heatmap=False,
        predict_endpoint_offset=False,
        predict_connectivity=False,
        snap_endpoint_to_atom_center_during_train=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_endpoint_queries = num_endpoint_queries or self.num_queries
        self.line_representation = line_representation
        self.endpoint_heatmap_stride = endpoint_heatmap_stride
        self.use_deformable_sampling = use_deformable_sampling
        self.predict_endpoint_heatmap = predict_endpoint_heatmap
        self.predict_endpoint_offset = predict_endpoint_offset
        self.predict_connectivity = predict_connectivity
        self.snap_endpoint_to_atom_center_during_train = snap_endpoint_to_atom_center_during_train


@register()
class EndpointLineHungarianMatcher(nn.Module):
    __share__ = ['use_focal_loss']

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_line = weight_dict.get('cost_line', 0.0)
        self.cost_endpoint = weight_dict.get('cost_endpoint', 0.0)
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets):
        bs, num_queries = outputs['pred_line_logits'].shape[:2]
        sizes = [len(v['line_labels']) for v in targets]

        if sum(sizes) == 0:
            empty = torch.zeros((0,), dtype=torch.int64)
            return {'indices': [(empty, empty) for _ in targets]}

        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs['pred_line_logits'].flatten(0, 1))
        else:
            out_prob = outputs['pred_line_logits'].flatten(0, 1).softmax(-1)

        out_lines = outputs['pred_lines'].flatten(0, 1)
        out_lines_swapped = _swap_line_endpoints(out_lines)
        tgt_ids = torch.cat([v['line_labels'] for v in targets])
        tgt_lines = torch.cat([v['lines'] for v in targets])

        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        cost_line = torch.minimum(
            torch.cdist(out_lines, tgt_lines, p=1),
            torch.cdist(out_lines_swapped, tgt_lines, p=1),
        )
        cost_endpoint = _line_endpoint_cost(out_lines, tgt_lines)
        C = self.cost_class * cost_class + self.cost_line * cost_line + self.cost_endpoint * cost_endpoint
        C = torch.nan_to_num(C.view(bs, num_queries, -1).cpu(), nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]
        return {'indices': indices}


@register()
class EndpointLineCriterion(nn.Module):
    __inject__ = ['matcher']

    def __init__(
        self,
        matcher,
        weight_dict,
        num_classes,
        losses=('focal', 'lines', 'endpoints', 'endpoint_atom'),
        alpha=0.25,
        gamma=2.0,
        relative_length_eps=1e-3,
        endpoint_positive_radius=4.0,
    ):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.num_classes = num_classes
        self.losses = list(losses)
        self.alpha = alpha
        self.gamma = gamma
        self.relative_length_eps = relative_length_eps
        self.endpoint_positive_radius = endpoint_positive_radius

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_line_focal(self, outputs, targets, indices, num_lines):
        src_logits = outputs['pred_line_logits']
        idx = self._get_src_permutation_idx(indices)
        if num_lines > 0:
            target_classes_o = torch.cat([t['line_labels'][J] for t, (_, J) in zip(targets, indices)])
        else:
            target_classes_o = torch.zeros((0,), dtype=torch.int64, device=src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        if len(target_classes_o) > 0:
            target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(src_logits.dtype)
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / max(num_lines, 1)
        return {'loss_line_focal': loss}

    def loss_lines(self, outputs, targets, indices, num_lines):
        if num_lines == 0:
            zero = outputs['pred_lines'].sum() * 0.0
            return {'loss_line_l1': zero, 'loss_line_rel': zero}

        idx = self._get_src_permutation_idx(indices)
        src_lines = outputs['pred_lines'][idx]
        target_lines = torch.cat([t['lines'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        direct = F.l1_loss(src_lines, target_lines, reduction='none').sum(-1)
        reverse = F.l1_loss(_swap_line_endpoints(src_lines), target_lines, reduction='none').sum(-1)
        abs_error = torch.minimum(direct, reverse)

        gt_lengths = torch.linalg.norm(target_lines[:, 2:4] - target_lines[:, 0:2], dim=-1)
        gt_lengths = gt_lengths.clamp(min=self.relative_length_eps)
        rel_error = abs_error / gt_lengths

        return {
            'loss_line_l1': abs_error.sum() / max(num_lines, 1),
            'loss_line_rel': rel_error.sum() / max(num_lines, 1),
        }

    def loss_endpoints(self, outputs, targets, indices, num_lines):
        if num_lines == 0:
            zero = outputs['pred_lines'].sum() * 0.0
            return {'loss_line_endpoint': zero}

        idx = self._get_src_permutation_idx(indices)
        src_lines = outputs['pred_lines'][idx]
        target_lines = torch.cat([t['lines'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        src_a, src_b = src_lines[:, :2], src_lines[:, 2:4]
        tgt_a, tgt_b = target_lines[:, :2], target_lines[:, 2:4]
        direct = torch.linalg.norm(src_a - tgt_a, dim=-1) + torch.linalg.norm(src_b - tgt_b, dim=-1)
        reverse = torch.linalg.norm(src_a - tgt_b, dim=-1) + torch.linalg.norm(src_b - tgt_a, dim=-1)
        endpoint_error = torch.minimum(direct, reverse)
        return {'loss_line_endpoint': endpoint_error.sum() / max(num_lines, 1)}

    def loss_endpoint_atom(self, outputs, line_targets, full_targets, indices, num_lines):
        if num_lines == 0:
            zero = outputs['pred_lines'].sum() * 0.0
            return {'loss_line_endpoint_atom': zero}

        idx = self._get_src_permutation_idx(indices)
        src_lines = outputs['pred_lines'][idx]
        assigned_centers = []
        for line_target, full_target, (_, tgt_idx) in zip(line_targets, full_targets, indices):
            if len(tgt_idx) == 0:
                continue
            target_lines = line_target['lines'][tgt_idx]
            centers = _target_box_centers(full_target)
            assigned_centers.append(_assign_line_to_atom_centers(target_lines, centers))

        if not assigned_centers:
            zero = outputs['pred_lines'].sum() * 0.0
            return {'loss_line_endpoint_atom': zero}

        target_centers = torch.cat(assigned_centers, dim=0)
        src_a, src_b = src_lines[:, :2], src_lines[:, 2:4]
        tgt_a, tgt_b = target_centers[:, :2], target_centers[:, 2:4]
        direct = F.l1_loss(src_a, tgt_a, reduction='none').sum(-1) + F.l1_loss(src_b, tgt_b, reduction='none').sum(-1)
        reverse = F.l1_loss(src_a, tgt_b, reduction='none').sum(-1) + F.l1_loss(src_b, tgt_a, reduction='none').sum(-1)
        endpoint_center_error = torch.minimum(direct, reverse)
        return {'loss_line_endpoint_atom': endpoint_center_error.sum() / max(num_lines, 1)}

    def _compute_losses(self, outputs, line_targets, full_targets):
        indices = self.matcher(outputs, line_targets)['indices']
        num_lines = sum(len(t['line_labels']) for t in line_targets)
        losses = {}
        if 'focal' in self.losses:
            losses.update(self.loss_line_focal(outputs, line_targets, indices, num_lines))
        if 'lines' in self.losses:
            losses.update(self.loss_lines(outputs, line_targets, indices, num_lines))
        if 'endpoints' in self.losses:
            losses.update(self.loss_endpoints(outputs, line_targets, indices, num_lines))
        if 'endpoint_atom' in self.losses:
            losses.update(self.loss_endpoint_atom(outputs, line_targets, full_targets, indices, num_lines))
        return {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}

    def forward(self, outputs, targets, **kwargs):
        if 'pred_line_logits' not in outputs:
            return {}

        line_targets = [{'lines': t['lines'], 'line_labels': t['line_labels']} for t in targets]
        losses = self._compute_losses(outputs, line_targets, targets)

        if 'line_aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['line_aux_outputs']):
                aux_losses = self._compute_losses(aux_outputs, line_targets, targets)
                aux_losses = {f'{k}_aux_{i}': v for k, v in aux_losses.items()}
                losses.update(aux_losses)

        return losses


@register()
class ChemEndpointCriterion(nn.Module):
    __inject__ = ['box_criterion', 'line_criterion']

    def __init__(self, box_criterion, line_criterion):
        super().__init__()
        self.box_criterion = box_criterion
        self.line_criterion = line_criterion

    def train(self, mode: bool = True):
        super().train(mode)
        self.box_criterion.train(mode)
        self.line_criterion.train(mode)
        return self

    def forward(self, outputs, targets, **kwargs):
        losses = {}
        losses.update(self.box_criterion(outputs, targets, **kwargs))
        losses.update(self.line_criterion(outputs, targets, **kwargs))
        return losses


@register()
class ChemEndpointPostProcessor(ChemPostProcessor):
    def __init__(
        self,
        box_postprocessor,
        line_num_classes,
        use_focal_loss=True,
        num_top_queries=300,
        line_num_top_queries=None,
        endpoint_score_thresh=0.15,
        snap_to_atom_centers=False,
        atom_center_snap_radius=24.0,
    ):
        super().__init__(
            box_postprocessor=box_postprocessor,
            line_num_classes=line_num_classes,
            use_focal_loss=use_focal_loss,
            num_top_queries=num_top_queries,
            line_num_top_queries=line_num_top_queries,
        )
        self.endpoint_score_thresh = endpoint_score_thresh
        self.snap_to_atom_centers = snap_to_atom_centers
        self.atom_center_snap_radius = atom_center_snap_radius
