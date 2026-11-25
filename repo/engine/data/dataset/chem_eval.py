import json
import os

import torch

from ...core import register


def _box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _line_distance_matrix(lines1, lines2):
    if lines1.numel() == 0 or lines2.numel() == 0:
        return torch.zeros((lines1.shape[0], lines2.shape[0]), dtype=torch.float32)
    direct = torch.cdist(lines1, lines2, p=2)
    reverse = torch.cdist(lines1[:, [2, 3, 0, 1]], lines2, p=2)
    return torch.minimum(direct, reverse) / 2.0


def _line_pairwise_endpoint_metrics(lines1, lines2):
    if lines1.numel() == 0 or lines2.numel() == 0:
        shape = (lines1.shape[0], lines2.shape[0])
        zeros = torch.zeros(shape, dtype=torch.float32)
        return zeros, zeros

    p1 = lines1[:, None, 0:2]
    p2 = lines1[:, None, 2:4]
    g1 = lines2[None, :, 0:2]
    g2 = lines2[None, :, 2:4]

    direct_d1 = torch.linalg.norm(p1 - g1, dim=-1)
    direct_d2 = torch.linalg.norm(p2 - g2, dim=-1)
    reverse_d1 = torch.linalg.norm(p1 - g2, dim=-1)
    reverse_d2 = torch.linalg.norm(p2 - g1, dim=-1)

    direct_mean = (direct_d1 + direct_d2) / 2.0
    reverse_mean = (reverse_d1 + reverse_d2) / 2.0
    endpoint_mean = torch.minimum(direct_mean, reverse_mean)

    gt_lengths = torch.linalg.norm(lines2[:, 2:4] - lines2[:, 0:2], dim=-1).clamp(min=1e-6)
    relative = endpoint_mean / gt_lengths[None, :]
    return endpoint_mean, relative


def _load_class_names(dataset):
    classes_path = os.path.join(os.path.dirname(dataset.atom_ann_file), 'classes.json')
    if not os.path.exists(classes_path):
        return {}, {}

    with open(classes_path, 'r') as f:
        class_data = json.load(f)

    def normalize_names(section):
        raw = class_data.get(section, {})
        return {int(k): str(v) for k, v in raw.items()}

    return normalize_names('atom_charge'), normalize_names('bond')


@register()
class ChemEvaluator(object):
    def __init__(self, dataset, box_iou_thresh=0.5, line_distance_thresh=12.0):
        self.dataset = dataset
        self.box_iou_thresh = box_iou_thresh
        self.line_distance_thresh = line_distance_thresh
        self.box_class_names, self.line_class_names = _load_class_names(dataset)
        self.iou_types = []
        self.coco_eval = {}
        self.cleanup()

    def cleanup(self):
        self.box_tp = 0
        self.box_fp = 0
        self.box_fn = 0
        self.box_iou_sum = 0.0
        self.box_class_stats = {}
        self.line_tp = 0
        self.line_fp = 0
        self.line_fn = 0
        self.line_dist_sum = 0.0
        self.line_endpoint_mean_sum = 0.0
        self.line_relative_dist_sum = 0.0
        self.line_class_stats = {}

    def _get_class_stat(self, stat_dict, class_id):
        class_id = int(class_id)
        if class_id not in stat_dict:
            stat_dict[class_id] = {'tp': 0, 'fp': 0, 'fn': 0}
        return stat_dict[class_id]

    def _record_box_tp(self, class_id):
        self._get_class_stat(self.box_class_stats, class_id)['tp'] += 1

    def _record_box_fp(self, class_id):
        self._get_class_stat(self.box_class_stats, class_id)['fp'] += 1

    def _record_box_fn(self, class_id):
        self._get_class_stat(self.box_class_stats, class_id)['fn'] += 1

    def _record_line_tp(self, class_id):
        self._get_class_stat(self.line_class_stats, class_id)['tp'] += 1

    def _record_line_fp(self, class_id):
        self._get_class_stat(self.line_class_stats, class_id)['fp'] += 1

    def _record_line_fn(self, class_id):
        self._get_class_stat(self.line_class_stats, class_id)['fn'] += 1

    def _match_boxes(self, pred, gt):
        pred_boxes = pred['boxes'].detach().cpu()
        pred_labels = pred['labels'].detach().cpu()
        pred_scores = pred['scores'].detach().cpu()
        gt_boxes = gt['boxes'].detach().cpu()
        gt_labels = gt['labels'].detach().cpu()

        if gt_boxes.numel() == 0:
            self.box_fp += pred_boxes.shape[0]
            for label in pred_labels.tolist():
                self._record_box_fp(label)
            return

        ious = _box_iou(pred_boxes, gt_boxes)
        used = set()
        order = torch.argsort(pred_scores, descending=True)
        for idx in order.tolist():
            same_cls = pred_labels[idx] == gt_labels
            valid = torch.where(same_cls & (ious[idx] >= self.box_iou_thresh))[0]
            if len(valid) == 0:
                self.box_fp += 1
                self._record_box_fp(pred_labels[idx])
                continue
            best = max(valid.tolist(), key=lambda j: float(ious[idx, j]) if j not in used else -1.0)
            if best in used or float(ious[idx, best]) < self.box_iou_thresh:
                self.box_fp += 1
                self._record_box_fp(pred_labels[idx])
                continue
            used.add(best)
            self.box_tp += 1
            self.box_iou_sum += float(ious[idx, best])
            self._record_box_tp(pred_labels[idx])

        for j in range(gt_boxes.shape[0]):
            if j not in used:
                self.box_fn += 1
                self._record_box_fn(gt_labels[j])

    def _match_lines(self, pred, gt):
        pred_lines = pred['lines'].detach().cpu()
        pred_labels = pred['line_labels'].detach().cpu()
        pred_scores = pred['line_scores'].detach().cpu()
        gt_lines = gt['lines'].detach().cpu()
        gt_labels = gt['line_labels'].detach().cpu()

        if gt_lines.numel() == 0:
            self.line_fp += pred_lines.shape[0]
            for label in pred_labels.tolist():
                self._record_line_fp(label)
            return

        dists = _line_distance_matrix(pred_lines, gt_lines)
        endpoint_means, relative_dists = _line_pairwise_endpoint_metrics(pred_lines, gt_lines)
        used = set()
        order = torch.argsort(pred_scores, descending=True)
        for idx in order.tolist():
            same_cls = pred_labels[idx] == gt_labels
            valid = torch.where(same_cls & (dists[idx] <= self.line_distance_thresh))[0]
            if len(valid) == 0:
                self.line_fp += 1
                self._record_line_fp(pred_labels[idx])
                continue
            best = min(valid.tolist(), key=lambda j: float(dists[idx, j]) if j not in used else 1e9)
            if best in used or float(dists[idx, best]) > self.line_distance_thresh:
                self.line_fp += 1
                self._record_line_fp(pred_labels[idx])
                continue
            used.add(best)
            self.line_tp += 1
            self.line_dist_sum += float(dists[idx, best])
            self.line_endpoint_mean_sum += float(endpoint_means[idx, best])
            self.line_relative_dist_sum += float(relative_dists[idx, best])
            self._record_line_tp(pred_labels[idx])

        for j in range(gt_lines.shape[0]):
            if j not in used:
                self.line_fn += 1
                self._record_line_fn(gt_labels[j])

    def update(self, res):
        for image_id, pred in res.items():
            gt = self.dataset.get_ground_truth(image_id)
            self._match_boxes(pred, gt)
            self._match_lines(pred, gt)

    def synchronize_between_processes(self):
        return

    def accumulate(self):
        return

    def _safe_prf(self, tp, fp, fn):
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        return precision, recall, f1

    def _summarize_class_stats(self, stat_dict, class_names):
        out = {}
        for class_id in sorted(stat_dict.keys()):
            stats = stat_dict[class_id]
            precision, recall, f1 = self._safe_prf(stats['tp'], stats['fp'], stats['fn'])
            out[class_names.get(class_id, str(class_id))] = {
                'id': class_id,
                'tp': stats['tp'],
                'fp': stats['fp'],
                'fn': stats['fn'],
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        return out

    def _format_worst_classes(self, class_summary, topk=5):
        if not class_summary:
            return 'n/a'
        items = sorted(
            class_summary.items(),
            key=lambda kv: (kv[1]['f1'], -(kv[1]['tp'] + kv[1]['fn'])),
        )[:topk]
        return ', '.join(f"{name}:F1={stats['f1']:.3f}" for name, stats in items)

    def summarize(self):
        box_precision, box_recall, box_f1 = self._safe_prf(self.box_tp, self.box_fp, self.box_fn)
        line_precision, line_recall, line_f1 = self._safe_prf(self.line_tp, self.line_fp, self.line_fn)
        box_iou = self.box_iou_sum / max(self.box_tp, 1)
        line_dist = self.line_dist_sum / max(self.line_tp, 1)
        line_endpoint_mean = self.line_endpoint_mean_sum / max(self.line_tp, 1)
        line_relative_dist = self.line_relative_dist_sum / max(self.line_tp, 1)
        box_per_class = self._summarize_class_stats(self.box_class_stats, self.box_class_names)
        line_per_class = self._summarize_class_stats(self.line_class_stats, self.line_class_names)
        print(
            f'ChemEval box: P={box_precision:.4f} R={box_recall:.4f} F1={box_f1:.4f} IoU={box_iou:.4f} | '
            f'line: P={line_precision:.4f} R={line_recall:.4f} F1={line_f1:.4f} Dist={line_dist:.4f} '
            f'EndErr={line_endpoint_mean:.4f} RelErr={line_relative_dist:.4f}'
        )
        print(f'ChemEval worst atom_charge classes: {self._format_worst_classes(box_per_class)}')
        print(f'ChemEval worst bond classes: {self._format_worst_classes(line_per_class)}')

    def get_stats(self):
        box_precision, box_recall, box_f1 = self._safe_prf(self.box_tp, self.box_fp, self.box_fn)
        line_precision, line_recall, line_f1 = self._safe_prf(self.line_tp, self.line_fp, self.line_fn)
        box_iou = self.box_iou_sum / max(self.box_tp, 1)
        line_dist = self.line_dist_sum / max(self.line_tp, 1)
        line_endpoint_mean = self.line_endpoint_mean_sum / max(self.line_tp, 1)
        line_relative_dist = self.line_relative_dist_sum / max(self.line_tp, 1)
        return {
            'chem_eval_box': [box_precision, box_recall, box_f1, box_iou],
            'chem_eval_line': [line_precision, line_recall, line_f1, line_dist],
            'chem_eval_line_v2': [line_precision, line_recall, line_f1, line_dist, line_endpoint_mean, line_relative_dist],
            'chem_eval_box_per_class': self._summarize_class_stats(self.box_class_stats, self.box_class_names),
            'chem_eval_line_per_class': self._summarize_class_stats(self.line_class_stats, self.line_class_names),
        }
