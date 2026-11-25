from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from scipy.optimize import linear_sum_assignment

from ..core import register


def _swap_line_endpoints(lines):
    return lines[..., [2, 3, 0, 1]]


@register()
class LineHungarianMatcher(nn.Module):
    __share__ = ['use_focal_loss']

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_line = weight_dict['cost_line']
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
        C = self.cost_class * cost_class + self.cost_line * cost_line
        C = torch.nan_to_num(C.view(bs, num_queries, -1).cpu(), nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]
        return {'indices': indices}


@register()
class LineCriterion(nn.Module):
    __inject__ = ['matcher']

    def __init__(
        self,
        matcher,
        weight_dict,
        num_classes,
        losses=('focal', 'lines'),
        alpha=0.25,
        gamma=2.0,
        relative_length_eps=1e-3,
    ):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.num_classes = num_classes
        self.losses = list(losses)
        self.alpha = alpha
        self.gamma = gamma
        self.relative_length_eps = relative_length_eps

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_line_focal(self, outputs, targets, indices, num_lines):
        src_logits = outputs['pred_line_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['line_labels'][J] for t, (_, J) in zip(targets, indices)]) if num_lines > 0 else torch.zeros((0,), dtype=torch.int64, device=src_logits.device)
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

    def _compute_losses(self, outputs, targets):
        indices = self.matcher(outputs, targets)['indices']
        num_lines = sum(len(t['line_labels']) for t in targets)
        losses = {}
        if 'focal' in self.losses:
            losses.update(self.loss_line_focal(outputs, targets, indices, num_lines))
        if 'lines' in self.losses:
            losses.update(self.loss_lines(outputs, targets, indices, num_lines))
        return {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}

    def forward(self, outputs, targets, **kwargs):
        if 'pred_line_logits' not in outputs:
            return {}

        line_targets = [{'lines': t['lines'], 'line_labels': t['line_labels']} for t in targets]
        losses = self._compute_losses(outputs, line_targets)

        if 'line_aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['line_aux_outputs']):
                aux_losses = self._compute_losses(aux_outputs, line_targets)
                aux_losses = {f'{k}_aux_{i}': v for k, v in aux_losses.items()}
                losses.update(aux_losses)

        if 'line_enc_aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['line_enc_aux_outputs']):
                aux_losses = self._compute_losses(aux_outputs, line_targets)
                aux_losses = {f'{k}_enc_{i}': v for k, v in aux_losses.items()}
                losses.update(aux_losses)

        return losses


@register()
class ChemCriterion(nn.Module):
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
