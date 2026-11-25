import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from .postprocessor import mod


@register()
class ChemPostProcessor(nn.Module):
    __inject__ = ['box_postprocessor']

    def __init__(
        self,
        box_postprocessor,
        line_num_classes,
        use_focal_loss=True,
        num_top_queries=300,
        line_num_top_queries=None,
    ):
        super().__init__()
        self.box_postprocessor = box_postprocessor
        self.line_num_classes = int(line_num_classes)
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.line_num_top_queries = line_num_top_queries or num_top_queries

    def forward(self, outputs, orig_target_sizes):
        box_results = self.box_postprocessor(outputs, orig_target_sizes)
        box_results_is_tuple = isinstance(box_results, tuple)
        if box_results_is_tuple:
            box_labels, box_boxes, box_scores = box_results

        logits = outputs['pred_line_logits']
        lines = outputs['pred_lines'] * orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.line_num_top_queries, dim=-1)
            labels = mod(index, self.line_num_classes)
            index = index // self.line_num_classes
            lines = lines.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, lines.shape[-1]))
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.line_num_top_queries:
                scores, index = torch.topk(scores, self.line_num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                lines = torch.gather(lines, dim=1, index=index.unsqueeze(-1).repeat(1, 1, lines.shape[-1]))

        results = []
        iterable = zip(labels, lines, scores)
        if box_results_is_tuple:
            iterable = zip(box_labels, box_boxes, box_scores, labels, lines, scores)
        else:
            iterable = zip(box_results, labels, lines, scores)

        for item in iterable:
            if box_results_is_tuple:
                box_label, box_box, box_score, line_label, line_value, line_score = item
                result = {
                    'labels': box_label,
                    'boxes': box_box,
                    'scores': box_score,
                }
            else:
                box_result, line_label, line_value, line_score = item
                result = dict(box_result)
            result['line_labels'] = line_label
            result['lines'] = line_value
            result['line_scores'] = line_score
            results.append(result)
        return results if not box_results_is_tuple else results

    def deploy(self):
        self.eval()
        if hasattr(self.box_postprocessor, 'deploy'):
            self.box_postprocessor.deploy()
        return self
