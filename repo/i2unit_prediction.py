#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

from engine.core import YAMLConfig
from engine.rtv4.chem_graph import assemble_molecular_graph, result_from_graph
from engine.rtv4.chem_ocr_expand import ocr_expand_graph
from engine.solver._solver import remove_module_prefix


DEFAULT_CONFIG = ROOT.parent / "config/train_mp123_full.yml"
DEFAULT_RESUME = ROOT.parent / "weights/checkpoint0119.pth"
DEFAULT_CLASSES = ROOT.parent / "dataset/annotations/classes.json"

BOX_COLORS = {
    "default": (0, 255, 0),
    "charge": (255, 0, 0),
}
BOND_COLORS = {
    0: (128, 128, 128),
    1: (0, 128, 255),
    2: (255, 0, 0),
    3: (255, 140, 0),
    4: (255, 0, 255),
    5: (255, 215, 0),
    6: (128, 0, 128),
}


def _box_center(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5]


def _line_center(line):
    x1, y1, x2, y2 = line
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5]


def _point_distance(a, b):
    ax, ay = a
    bx, by = b
    return float(np.hypot(ax - bx, ay - by))


def _nearest_center_index(point, centers):
    if not centers:
        raise ValueError("centers must not be empty")
    points = np.asarray(centers, dtype=np.float32)
    point_arr = np.asarray(point, dtype=np.float32)
    dists = np.sum((points - point_arr) ** 2, axis=1)
    return int(np.argmin(dists))


def _build_center_lookup(centers):
    points = np.asarray(centers, dtype=np.float32)
    tree = KDTree(points) if KDTree is not None and len(points) > 0 else None
    return tree, points


def _query_nearest_center_index(point, tree, points):
    if len(points) == 0:
        raise ValueError("points must not be empty")
    if tree is not None:
        _, idx = tree.query(np.asarray(point, dtype=np.float32), k=1)
        return int(idx)
    return _nearest_center_index(point, points.tolist())


def _pick_font(size=18):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    return remove_module_prefix(state)


def _build_model(config_path, checkpoint_path, device):
    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    state = _load_checkpoint(checkpoint_path)
    cfg.model.load_state_dict(state, strict=False)
    model = cfg.model.deploy().to(device).eval()
    postprocessor = cfg.postprocessor.deploy().to(device).eval()
    return cfg, model, postprocessor


def _load_classes(path):
    with open(path, "r") as f:
        data = json.load(f)
    box_classes = {int(k): str(v) for k, v in data.get("atom_charge", {}).items()}
    line_classes = {int(k): str(v) for k, v in data.get("bond", {}).items()}
    return box_classes, line_classes


def _infer_runtime_from_config(cfg):
    val_dataset = cfg.yaml_cfg.get("val_dataloader", {}).get("dataset", {})
    train_dataset = cfg.yaml_cfg.get("train_dataloader", {}).get("dataset", {})
    dataset_cfg = val_dataset or train_dataset

    image_size = tuple(dataset_cfg.get("image_size", cfg.yaml_cfg.get("eval_spatial_size", [640, 640])))
    mean = tuple(dataset_cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(dataset_cfg.get("std", (0.229, 0.224, 0.225)))

    class_path = None
    atom_ann_file = dataset_cfg.get("atom_ann_file")
    if atom_ann_file:
        candidate = Path(atom_ann_file).with_name("classes.json")
        if candidate.exists():
            class_path = candidate

    search_dirs = []
    for ds_cfg in (val_dataset, train_dataset):
        folder = ds_cfg.get("img_folder")
        if folder:
            search_dirs.append(Path(folder))

    return image_size, mean, std, class_path, search_dirs


def _resolve_image_path(image_arg, search_dirs):
    image_path = Path(image_arg)
    if image_path.exists():
        return image_path.resolve()

    for base in search_dirs:
        candidate = base / image_arg
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Image not found: {image_arg}")


def _resize_and_normalize(image, image_size, mean, std):
    image = image.resize(tuple(image_size), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))
    tensor = tensor.permute(2, 0, 1) / 255.0
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def _scale_result(result, src_size, dst_size):
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    sx = dst_w / max(float(src_w), 1.0)
    sy = dst_h / max(float(src_h), 1.0)

    scaled = {}
    for key, value in result.items():
        scaled[key] = value.detach().cpu().clone() if hasattr(value, "detach") else value

    if "boxes" in scaled:
        scaled["boxes"] = scaled["boxes"] * torch.tensor([sx, sy, sx, sy], dtype=torch.float32)
    if "lines" in scaled:
        scaled["lines"] = scaled["lines"] * torch.tensor([sx, sy, sx, sy], dtype=torch.float32)
    return scaled


def _draw_label(draw, xy, text, color):
    x, y = xy
    bbox = draw.textbbox((x, y), text)
    draw.rectangle(bbox, fill=(255, 255, 255))
    draw.text((x, y), text, fill=color)


def _draw_annotations(image, result, box_classes, line_classes, title, score_thr=None):
    draw = ImageDraw.Draw(image)
    title_font = _pick_font(22)
    draw.rectangle((0, 0, image.size[0], 34), fill=(255, 255, 255))
    draw.text((10, 6), title, fill=(20, 20, 20), font=title_font)

    scores = result.get("scores")
    if scores is None:
        scores = [None] * len(result.get("labels", []))

    for score, label, box in zip(scores, result.get("labels", []), result.get("boxes", [])):
        score = None if score is None else float(score)
        label = int(label)
        if score_thr is not None and score is not None and score < score_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        class_name = box_classes.get(label, str(label))
        color = BOX_COLORS["charge"] if class_name.startswith("+") or class_name.startswith("-") else BOX_COLORS["default"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        text = class_name if score is None else f"{class_name} {score:.2f}"
        _draw_label(draw, (x1, max(36, y1 - 14)), text, color)

    line_scores = result.get("line_scores")
    if line_scores is None:
        line_scores = [None] * len(result.get("line_labels", []))

    for score, label, line in zip(line_scores, result.get("line_labels", []), result.get("lines", [])):
        score = None if score is None else float(score)
        label = int(label)
        if score_thr is not None and score is not None and score < score_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in line]
        color = BOND_COLORS.get(label, (255, 255, 255))
        class_name = line_classes.get(label, str(label))
        draw.line((x1, y1, x2, y2), fill=color, width=3)
        text = class_name if score is None else f"{class_name} {score:.2f}"
        _draw_label(draw, (((x1 + x2) / 2.0), max(36, (y1 + y2) / 2.0)), text, color)

    return image


def _draw_structured_raw(image, filtered_boxes, kept_lines, title):
    draw = ImageDraw.Draw(image)
    title_font = _pick_font(22)
    draw.rectangle((0, 0, image.size[0], 34), fill=(255, 255, 255))
    draw.text((10, 6), title, fill=(20, 20, 20), font=title_font)

    for item in filtered_boxes:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        class_name = item["label_name"]
        color = BOX_COLORS["charge"] if class_name.startswith("+") or class_name.startswith("-") else BOX_COLORS["default"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        text = f"A{item['atom_index']} {class_name} {item['score']:.2f}"
        _draw_label(draw, (x1, max(36, y1 - 14)), text, color)

    for item in kept_lines:
        x1, y1, x2, y2 = item["line_xyxy"]
        color = BOND_COLORS.get(item["label_id"], (255, 255, 255))
        draw.line((x1, y1, x2, y2), fill=color, width=3)
        a, b = item["atom_pair"]
        text = f"B{item['bond_index']} A{a}-A{b} {item['label_name']} {item['score']:.2f}"
        _draw_label(draw, (item["line_center_xy"][0], max(36, item["line_center_xy"][1])), text, color)

    return image


def _is_charge_label(label_name):
    return label_name.startswith("+") or label_name.startswith("-")


def _filter_sort_and_nms_boxes(result, box_classes, score_thr, nms_iou_thr):
    candidates = []
    scores = result.get("scores")
    if scores is None:
        scores = [None] * len(result.get("labels", []))

    for raw_index, (score, label, box) in enumerate(zip(scores, result.get("labels", []), result.get("boxes", []))):
        score = None if score is None else float(score)
        label = int(label)
        if score_thr is not None and score is not None and score < score_thr:
            continue

        box = [float(v) for v in box]
        label_name = box_classes.get(label, str(label))
        if _is_charge_label(label_name):
            continue
        center = _box_center(box)
        candidates.append(
            {
                "raw_index": raw_index,
                "label_id": label,
                "label_name": label_name,
                "score": score,
                "bbox_xyxy": box,
                "center_xy": center,
            }
        )

    if not candidates:
        return []

    boxes_tensor = torch.tensor([item["bbox_xyxy"] for item in candidates], dtype=torch.float32)
    scores_tensor = torch.tensor([float(item["score"] or 0.0) for item in candidates], dtype=torch.float32)
    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, nms_iou_thr).tolist()
    filtered_boxes = [candidates[idx] for idx in keep_indices]

    filtered_boxes.sort(key=lambda item: (item["center_xy"][0], item["center_xy"][1], item["raw_index"]))
    for atom_index, item in enumerate(filtered_boxes):
        item["atom_index"] = atom_index
    return filtered_boxes


def _dedupe_lines_by_nearest_boxes(
    result, filtered_boxes, line_classes, line_score_thr,
    weak_endpoint_ratio=0.15, weak_score_cutoff=0.4,
    image_evidence=None,
):
    """Keep one score-winning bond class per atom pair, preserving endpoint order.

    A geometry gate applies only to weak candidates: both endpoints
    must lie near their assigned atom centers relative to the pair's length.
    Crossing lines with different atom pairs remain separate bonds. Pass None
    for weak_endpoint_ratio to explicitly disable the gate for comparisons.
    """
    if not filtered_boxes:
        return []

    centers = [item["center_xy"] for item in filtered_boxes]
    tree, points = _build_center_lookup(centers)
    best_by_pair = {}

    line_scores = result.get("line_scores")
    if line_scores is None:
        line_scores = [None] * len(result.get("line_labels", []))

    for raw_index, (score, label, line) in enumerate(zip(line_scores, result.get("line_labels", []), result.get("lines", []))):
        score = None if score is None else float(score)
        if score is not None and score < line_score_thr:
            continue
        label = int(label)
        line = [float(v) for v in line]
        begin = line[:2]
        end = line[2:4]

        begin_idx = _query_nearest_center_index(begin, tree, points)
        end_idx = _query_nearest_center_index(end, tree, points)
        if begin_idx == end_idx:
            continue

        begin_atom = filtered_boxes[begin_idx]
        end_atom = filtered_boxes[end_idx]
        direct_begin_dist = _point_distance(begin, begin_atom["center_xy"])
        direct_end_dist = _point_distance(end, end_atom["center_xy"])
        swap_begin_dist = _point_distance(begin, end_atom["center_xy"])
        swap_end_dist = _point_distance(end, begin_atom["center_xy"])

        direct_sum = direct_begin_dist + direct_end_dist
        swap_sum = swap_begin_dist + swap_end_dist
        if swap_sum < direct_sum:
            source_atom = end_atom
            target_atom = begin_atom
            source_dist = swap_begin_dist
            target_dist = swap_end_dist
            endpoint_distance_sum = swap_sum
        else:
            source_atom = begin_atom
            target_atom = end_atom
            source_dist = direct_begin_dist
            target_dist = direct_end_dist
            endpoint_distance_sum = direct_sum

        atom_pair = tuple(sorted((source_atom["atom_index"], target_atom["atom_index"])))
        if weak_endpoint_ratio is not None and (score is None or score < weak_score_cutoff):
            pair_length = _point_distance(source_atom['center_xy'], target_atom['center_xy'])
            endpoint_limit = max(2.0, float(weak_endpoint_ratio) * pair_length)
            if max(source_dist, target_dist) > endpoint_limit:
                continue
        pixel_support = None
        if image_evidence is not None:
            pixel_support = image_evidence.evaluate(line,label,source_atom,target_atom)
            if not pixel_support['accepted']:
                continue

        item = {
            "raw_index": raw_index,
            "label_id": label,
            "label_name": line_classes.get(label, str(label)),
            "score": score,
            "line_xyxy": line,
            "line_center_xy": _line_center(line),
            "source_atom_index": source_atom["atom_index"],
            "target_atom_index": target_atom["atom_index"],
            "atom_pair": list(atom_pair),
            "source_atom_center_xy": source_atom["center_xy"],
            "target_atom_center_xy": target_atom["center_xy"],
            "source_endpoint_xy": begin,
            "target_endpoint_xy": end,
            "source_endpoint_to_atom_center_distance": source_dist,
            "target_endpoint_to_atom_center_distance": target_dist,
            "endpoint_distance_sum": endpoint_distance_sum,
        }
        if pixel_support is not None:
            item['image_evidence'] = pixel_support

        best = best_by_pair.get(atom_pair)
        item_score = float("-inf") if score is None else float(score)
        if best is None:
            best_by_pair[atom_pair] = item
            continue

        best_dist = float(best["endpoint_distance_sum"])
        best_score = float("-inf") if best["score"] is None else float(best["score"])
        if (-item_score, endpoint_distance_sum, raw_index) < (-best_score, best_dist, best["raw_index"]):
            best_by_pair[atom_pair] = item

    kept_lines = sorted(
        best_by_pair.values(),
        key=lambda item: (
            item["atom_pair"][0],
            item["atom_pair"][1],
            item["endpoint_distance_sum"],
            -float(item["score"] or -1.0),
        ),
    )
    for bond_index, item in enumerate(kept_lines):
        item["bond_index"] = bond_index
    return kept_lines


def _build_structured_raw_result(filtered_boxes, kept_lines):
    result = {
        "boxes": torch.tensor([item["bbox_xyxy"] for item in filtered_boxes], dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor([item["label_id"] for item in filtered_boxes], dtype=torch.int64),
        "scores": torch.tensor([float(item["score"] or 0.0) for item in filtered_boxes], dtype=torch.float32),
        "lines": torch.tensor([item["line_xyxy"] for item in kept_lines], dtype=torch.float32).reshape(-1, 4),
        "line_labels": torch.tensor([item["label_id"] for item in kept_lines], dtype=torch.int64),
        "line_scores": torch.tensor([float(item["score"] or 0.0) for item in kept_lines], dtype=torch.float32),
    }
    if all('source_atom_index' in item and 'target_atom_index' in item for item in kept_lines):
        rows={item.get('atom_index',row):row for row,item in enumerate(filtered_boxes)
              if not _is_charge_label(item.get('label_name',''))}
        result['line_atom_indices']=torch.tensor(
            [[rows[item['source_atom_index']],rows[item['target_atom_index']]] for item in kept_lines],
            dtype=torch.int64).reshape(-1,2)
    return result


def _serialize_result(result, box_classes, line_classes, score_thr=None):
    payload = {"boxes": [], "lines": []}

    scores = result.get("scores")
    if scores is None:
        scores = [None] * len(result.get("labels", []))

    for score, label, box in zip(scores, result.get("labels", []), result.get("boxes", [])):
        score = None if score is None else float(score)
        label = int(label)
        if score_thr is not None and score is not None and score < score_thr:
            continue
        payload["boxes"].append(
            {
                "label_id": label,
                "label_name": box_classes.get(label, str(label)),
                "score": score,
                "bbox_xyxy": [float(v) for v in box],
            }
        )

    line_scores = result.get("line_scores")
    if line_scores is None:
        line_scores = [None] * len(result.get("line_labels", []))

    for score, label, line in zip(line_scores, result.get("line_labels", []), result.get("lines", [])):
        score = None if score is None else float(score)
        label = int(label)
        if score_thr is not None and score is not None and score < score_thr:
            continue
        payload["lines"].append(
            {
                "label_id": label,
                "label_name": line_classes.get(label, str(label)),
                "score": score,
                "line_xyxy": [float(v) for v in line],
            }
        )

    return payload


def _serialize_structured_raw(filtered_boxes, kept_lines, raw_result, threshold, line_score_threshold, box_nms_iou_threshold):
    return {
        "meta": {
            "mode": "raw_structured",
            "box_threshold": threshold,
            "line_score_threshold": line_score_threshold,
            "box_nms_iou_threshold": box_nms_iou_threshold,
            "raw_box_candidates": len(result_get(raw_result, "boxes")),
            "raw_line_candidates": len(result_get(raw_result, "lines")),
            "filtered_atom_box_count": len(filtered_boxes),
            "kept_line_count": len(kept_lines),
            "line_rule": "keep raw top300 lines with score >= threshold, map both endpoints to nearest filtered atom-box centers, require two different atoms, keep the line with the smallest endpoint-to-center distance sum per atom pair",
            "box_sort_rule": "sort by center_x then center_y ascending",
            "box_filter_rule": "keep non-charge atom boxes with score >= threshold, then apply NMS and keep higher-score box when IoU > 0.55",
            "nearest_search": "KDTree when scipy is available, otherwise brute-force nearest center",
        },
        "atoms": filtered_boxes,
        "bonds": kept_lines,
    }


def result_get(result, key):
    value = result.get(key, [])
    return value if value is not None else []


def _draw_ocr_labels(image, expansion):
    """Draw OCR-recognized labels on atom boxes."""
    draw = ImageDraw.Draw(image)
    font = _pick_font(16)
    color = (0, 0, 200)
    for info in expansion.get("ocr_atoms", []):
        box = info.get("box")
        text = info.get("text", "")
        if box and text:
            x1, y1, x2, y2 = box
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            _draw_label(draw, (x1, max(36, y1 - 16)), f"OCR:{text}", color)
    return image


@torch.no_grad()
def _infer_single(model, postprocessor, image_path, device, image_size, mean, std):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    tensor = _resize_and_normalize(image, image_size, mean, std).unsqueeze(0).to(device)
    scaled_size = torch.tensor([[image_size[0], image_size[1]]], device=device)
    outputs = model(tensor)
    raw_result = postprocessor(outputs, scaled_size)[0]
    raw_result = _scale_result(raw_result, image_size, (orig_w, orig_h))
    return image, raw_result


def main():
    parser = argparse.ArgumentParser(description="Single-image prediction for Chem RT-DETRv4.")
    parser.add_argument("-i", "--input", required=True, help="Image path or file name like 6346.png")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("-r", "--resume", default=str(DEFAULT_RESUME))
    parser.add_argument("-o", "--output-dir", default="single_prediction_output")
    parser.add_argument("--classes", default=None)
    parser.add_argument("--threshold", type=float, default=0.35, help="Score threshold used in raw mode")
    parser.add_argument("--box-nms-iou-thresh", type=float, default=0.55, help="NMS IoU threshold for raw atom boxes")
    parser.add_argument("--raw-line-threshold", type=float, default=0.35, help="Line score threshold used before nearest-atom pairing in raw mode")
    parser.add_argument("--mode", choices=["raw", "graph", "ocr_expand"], default="raw")
    parser.add_argument("--graph-box-score-thresh", type=float, default=0.5)
    parser.add_argument("--graph-line-score-thresh", type=float, default=0.35)
    parser.add_argument("-d", "--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.resume).resolve()

    cfg, model, postprocessor = _build_model(config_path, checkpoint_path, device)
    image_size, mean, std, inferred_classes_path, search_dirs = _infer_runtime_from_config(cfg)

    classes_path = Path(args.classes).resolve() if args.classes else inferred_classes_path or DEFAULT_CLASSES
    box_classes, line_classes = _load_classes(classes_path)

    image_path = _resolve_image_path(args.input, search_dirs)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image, raw_result = _infer_single(model, postprocessor, image_path, device, image_size, mean, std)

    if args.mode == "graph" or args.mode == "ocr_expand":
        graph = assemble_molecular_graph(
            raw_result,
            classes_path=str(classes_path),
            box_score_thresh=args.graph_box_score_thresh,
            line_score_thresh=args.graph_line_score_thresh,
        )
        final_result = result_from_graph(graph)
        final_result = _scale_result(final_result, image.size, image.size)
        title = f"Prediction (box>={args.graph_box_score_thresh:.2f}, line>={args.graph_line_score_thresh:.2f})"
        score_thr = None
        json_payload = graph

        if args.mode == "ocr_expand":
            expansion = ocr_expand_graph(graph, image)
            graph["ocr_expansion"] = expansion
            json_payload = graph
            title += " + OCR Expand"
            if expansion.get("expanded_smiles"):
                print(f"expanded_smiles: {expansion['expanded_smiles']}")
            if expansion.get("skeleton_smiles"):
                print(f"skeleton_smiles: {expansion['skeleton_smiles']}")
            if expansion.get("ocr_atoms"):
                for info in expansion["ocr_atoms"]:
                    print(f"  atom {info['atom_index']}: OCR='{info['text']}' -> SMILES='{info['smiles']}'")
    else:
        filtered_boxes = _filter_sort_and_nms_boxes(raw_result, box_classes, args.threshold, args.box_nms_iou_thresh)
        kept_lines = _dedupe_lines_by_nearest_boxes(raw_result, filtered_boxes, line_classes, args.raw_line_threshold)
        final_result = _build_structured_raw_result(filtered_boxes, kept_lines)
        title = f"Raw Structured Prediction (box>={args.threshold:.2f}, NMS IoU<={args.box_nms_iou_thresh:.2f}, line>={args.raw_line_threshold:.2f})"
        score_thr = None
        json_payload = _serialize_structured_raw(
            filtered_boxes,
            kept_lines,
            raw_result,
            args.threshold,
            args.raw_line_threshold,
            args.box_nms_iou_thresh,
        )
    
    if args.mode == "raw":
        vis = _draw_structured_raw(image.copy(), filtered_boxes, kept_lines, title=title)
    else:
        vis = _draw_annotations(image.copy(), final_result, box_classes, line_classes, title=title, score_thr=score_thr)
        # Draw OCR labels on atoms for ocr_expand mode
        if args.mode == "ocr_expand" and "ocr_expansion" in json_payload:
            vis = _draw_ocr_labels(vis, json_payload["ocr_expansion"])

    stem = image_path.stem
    vis_path = output_dir / f"{stem}_pred.png"
    json_path = output_dir / f"{stem}_pred.json"
    vis.save(vis_path)
    with open(json_path, "w") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    print(f"input_image: {image_path}")
    print(f"mode: {args.mode}")
    print(f"classes: {classes_path}")
    print(f"visualization: {vis_path}")
    print(f"prediction_json: {json_path}")


if __name__ == "__main__":
    main()
