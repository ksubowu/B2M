#!/usr/bin/env python3
"""Fine-tune the active detector, independent line decoder, and bond-pair head on Fix data.

The training split must contain only WebApp human Fix records. A separate
fresh-holdout split is used for evaluation and never contributes gradients.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert, box_iou, nms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "repo"), str(ROOT / "repo/i2bl_webapp")]
from engine.core import YAMLConfig
from pair_training.model import (PAIR_CLASSES, encode_bond, knn_pairs,
                                 predicted_atoms, run_pair_head, state_hash)
from pair_training.train import load_model, evaluate as evaluate_pairs

CHARGE_ID = {"-4": 13, "-2": 14, "-1": 15, "+1": 16, "+2": 17,
             "2": 17, "1": 16}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class JointHumanDataset(Dataset):
    def __init__(self, root, split, limit=0, lowres_probability=0.0):
        self.root = Path(root)
        self.records = [r for r in read_json(self.root / "manifest.json")
                        if r.get("split") == split]
        if split == "train":
            self.records = [r for r in self.records if r.get("source") == "webapp_human_fix"]
        elif split == "val":
            self.records = [r for r in self.records if r.get("source") == "external_fresh_validation"]
        if limit:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f"No {split} records in {root}")
        self.lowres_probability = lowres_probability

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        truth = read_json(self.root / row["truth"])
        width, height = map(int, truth["image_size"])
        with Image.open(self.root / row["image"]) as handle:
            image_native = handle.convert("RGB")
        if image_native.size != (width, height):
            raise ValueError(f"Truth/image size mismatch: {row['file_name']}")
        image = image_native
        resolution = 0
        if self.lowres_probability and random.random() < self.lowres_probability:
            resolution = random.choice([192, 224, 256, 288, 320])
            image = image.resize((resolution, resolution), Image.Resampling.BOX).convert("L")
            image = image.point(lambda value: 255 if value > 245 else 0).convert("RGB")
        tensor = TF.resize(TF.to_tensor(image), [640, 640])
        tensor = TF.normalize(tensor, [.485, .456, .406], [.229, .224, .225])

        labels, boxes = [], []
        centers, atom_labels, atom_boxes = [], [], []
        atoms = truth.get("atoms", [])
        for atom in atoms:
            cid = int(atom["category_id"])
            if not 0 <= cid <= 12:
                raise ValueError(f"Invalid atom class {cid}: {row['file_name']}")
            x, y, w, h = map(float, atom["bbox"])
            labels.append(cid)
            boxes.append([x / width, y / height, (x + w) / width, (y + h) / height])
            atom_boxes.append([x / width, y / height, (x + w) / width, (y + h) / height])
            cx, cy = map(float, atom["center"])
            centers.append([cx / width, cy / height])
            atom_labels.append(cid)
        for charge in truth.get("charges", []):
            name = str(charge.get("canonical_label", charge.get("label", ""))).strip()
            cid = CHARGE_ID.get(name)
            if cid is None:
                raise ValueError(f"Unsupported charge label {name!r}: {row['file_name']}")
            x1, y1, x2, y2 = map(float, charge["bbox"])
            labels.append(cid)
            boxes.append([x1 / width, y1 / height, x2 / width, y2 / height])
        if len(labels) > 300:
            raise ValueError(f"More than 300 atom/charge boxes: {row['file_name']} ({len(labels)})")

        edges = {}
        ids = {int(atom["index"]): i for i, atom in enumerate(atoms)}
        lengths = []
        for bond in truth.get("bonds", []):
            i, j = ids[int(bond["begin"])], ids[int(bond["end"])]
            category = int(bond["category_id"])
            if (i, j) in edges or (j, i) in edges:
                raise ValueError(f"Duplicate atom pair in Fix labels: {row['file_name']}")
            edges[i, j] = category
            lengths.append(float(np.linalg.norm(np.asarray(centers[i]) - np.asarray(centers[j]))))
        tolerance = max(4 / width, .25 * float(np.median(lengths))) if lengths else 4 / width
        line_targets = []
        for bond in truth.get("bonds", []):
            x1, y1, x2, y2 = map(float, bond["line"])
            line_targets.append([x1 / width, y1 / height, x2 / width, y2 / height])
        target = {
            "detection": {"labels": torch.tensor(labels, dtype=torch.long),
                          "boxes": box_convert(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                                               "xyxy", "cxcywh")},
            "centers": torch.tensor(centers, dtype=torch.float32),
            "labels": torch.tensor(atom_labels, dtype=torch.long), "edges": edges,
            "line_targets": {"lines": torch.tensor(line_targets, dtype=torch.float32).reshape(-1, 4),
                             "line_labels": torch.tensor([int(b["category_id"]) for b in truth.get("bonds", [])], dtype=torch.long),
                             "boxes": box_convert(torch.tensor(atom_boxes, dtype=torch.float32).reshape(-1, 4),
                                                  "xyxy", "cxcywh")},
            "tolerance": tolerance, "name": row["file_name"], "group": row.get("group"),
            "image_size": [width, height], "raster_resolution": resolution or min(width, height),
        }
        return tensor, target


def collate(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]


def detect_metrics(model, loader, device, score_threshold=.3):
    model.eval()
    totals = {"atom": [0, 0, 0], "charge": [0, 0, 0]}
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            _, outputs = model.extract(images)
            probs = outputs["pred_logits"].sigmoid()
            for bi, target in enumerate(targets):
                scores, labels = probs[bi].max(-1)
                boxes = outputs["pred_boxes"][bi]
                keep = scores >= score_threshold
                indices = torch.where(keep)[0]
                if len(indices):
                    kept = []
                    for cid in labels[indices].unique().tolist():
                        same = indices[labels[indices] == cid]
                        kept.extend(same[nms(box_convert(boxes[same], "cxcywh", "xyxy"),
                                             scores[same], .5)].tolist())
                    indices = torch.tensor(kept, device=device, dtype=torch.long)
                gt_boxes = target["detection"]["boxes"].to(device)
                gt_labels = target["detection"]["labels"].to(device)
                pred_boxes = boxes[indices]
                pred_labels = labels[indices]
                overlaps = box_iou(box_convert(pred_boxes, "cxcywh", "xyxy"),
                                   box_convert(gt_boxes, "cxcywh", "xyxy"))
                valid = (overlaps >= .5) & (pred_labels[:, None] == gt_labels[None, :])
                if valid.numel():
                    ii, jj = linear_sum_assignment(torch.where(valid, overlaps, -1).cpu().numpy(), maximize=True)
                    matches = [(a, b) for a, b in zip(ii, jj) if bool(valid[a, b])]
                else:
                    matches = []
                for name, predicate in (("atom", lambda x: x < 13), ("charge", lambda x: x >= 13)):
                    tp = sum(bool(predicate(int(gt_labels[j]))) for _, j in matches)
                    pred_count = sum(bool(predicate(int(x))) for x in pred_labels.tolist())
                    gt_count = sum(bool(predicate(int(x))) for x in gt_labels.tolist())
                    totals[name][0] += tp
                    totals[name][1] += pred_count
                    totals[name][2] += gt_count
    result = {}
    for name, (tp, pred, gt) in totals.items():
        result[name] = {"tp": tp, "predicted": pred, "truth": gt,
                        "precision": tp / max(1, pred), "recall": tp / max(1, gt),
                        "f1": 2 * tp / max(1, pred + gt)}
    return result


@torch.no_grad()
def line_metrics(model, loader, device, score_threshold=.3, endpoint_tolerance=16 / 640):
    """Evaluate the independent line detector without atom-pair proposals."""
    model.eval()
    tp = predicted = truth = exact = images_seen = 0
    for images, targets in loader:
        images = images.to(device)
        features = model.base.encoder(model.base.backbone(images))
        if isinstance(features, tuple):
            features = features[0]
        raw = model.base.line_decoder(features)
        probs = raw["pred_logits"].sigmoid()
        lines = raw["pred_boxes"]
        for i, target in enumerate(targets):
            flat = probs[i].flatten()
            scores, flat_idx = flat.topk(min(300, flat.numel()))
            keep = scores >= score_threshold
            scores, flat_idx = scores[keep], flat_idx[keep]
            query, labels = torch.div(flat_idx, probs.shape[-1], rounding_mode="floor"), flat_idx % probs.shape[-1]
            pred_lines = lines[i, query]
            gt = target["line_targets"]
            gt_lines, gt_labels = gt["lines"].to(device), gt["line_labels"].to(device)
            predicted += len(scores); truth += len(gt_lines); images_seen += 1
            image_tp = 0
            if len(pred_lines) and len(gt_lines):
                direct = torch.cdist(pred_lines[:, :2], gt_lines[:, :2]) + torch.cdist(pred_lines[:, 2:], gt_lines[:, 2:])
                reverse = torch.cdist(pred_lines[:, :2], gt_lines[:, 2:]) + torch.cdist(pred_lines[:, 2:], gt_lines[:, :2])
                endpoint_error = torch.minimum(direct, reverse) * .5
                class_ok = labels[:, None] == gt_labels[None, :]
                valid = class_ok & (endpoint_error <= endpoint_tolerance)
                cost = endpoint_error + (~class_ok).float() * 100 + (~(endpoint_error <= endpoint_tolerance)).float() * 10
                rows, cols = linear_sum_assignment(cost.cpu().numpy())
                image_tp = sum(bool(valid[r, c]) for r, c in zip(rows, cols))
            tp += image_tp
            exact += int(image_tp == len(scores) == len(gt_lines))
    precision = tp / max(1, predicted)
    recall = tp / max(1, truth)
    return {"tp": tp, "predicted": predicted, "truth": truth,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "exact_images": exact, "images": images_seen,
            "exact_image_rate": exact / max(1, images_seen),
            "score_threshold": score_threshold, "endpoint_tolerance_normalized": endpoint_tolerance}


def box_target_to_device(target, device):
    return {"labels": target["detection"]["labels"].to(device),
            "boxes": target["detection"]["boxes"].to(device)}


def run_pair_loss(model, features, outputs, images, targets, k, predicted_fraction,
                  score_mode="existence"):
    atom_rows, pairs_rows, label_rows = [], [], []
    for i, target in enumerate(targets):
        if random.random() < predicted_fraction:
            atom = predicted_atoms(outputs, i, cutoff=.3)
        else:
            atom = {"centers": target["centers"].to(images.device),
                    "labels": target["labels"].to(images.device)}
        pairs = knn_pairs(atom["centers"], k)
        labels = []
        # Match detected/GT centers to the human topology. Keep unmatched
        # predicted atoms out of the pair loss rather than treating them as negatives.
        from pair_training.data import pair_targets
        labels.append(pair_targets(atom, pairs, target, ignore_unmatched=True)[0])
        atom_rows.append(atom); pairs_rows.append(pairs); label_rows.append(labels[0])
    logits = run_pair_head(model.pair_head, features, atom_rows, pairs_rows, images)
    # Normalize within each image before averaging across images. Otherwise a
    # 156-atom drawing contributes ~25x more candidate pairs than a 7-atom
    # drawing and can dominate this tiny, source-limited dataset.
    image_losses, positives, negatives = [], 0, 0
    for image_logits, image_labels in zip(logits, label_rows):
        valid = image_labels != -100
        if not valid.any():
            continue
        supervised = image_logits[valid]
        labels = image_labels[valid]
        positive = labels > 0
        positives += int(positive.sum())
        negatives += int((labels == 0).sum())
        if score_mode == "existence":
            existence = torch.logsumexp(supervised[:, 1:], -1) - supervised[:, 0]
            image_loss = F.binary_cross_entropy_with_logits(existence, positive.float())
            if positive.any():
                image_loss = image_loss + F.cross_entropy(
                    supervised[positive, 1:], labels[positive] - 1)
        else:
            image_loss = F.cross_entropy(supervised, labels)
        image_losses.append(image_loss)
    if not image_losses:
        differentiable_zero = sum((x.sum() for x in logits), images.sum() * 0) * 0
        return differentiable_zero, 0, 0
    return torch.stack(image_losses).mean(), positives, negatives


def run_line_loss(model, features, targets, criterion):
    line_targets = [target["line_targets"] for target in targets]
    raw = model.base.line_decoder(features, targets=[{
        "labels": target["line_labels"], "boxes": target["lines"]
    } for target in line_targets])
    from engine.rtv4.chem_model import _remap_line_outputs
    outputs = _remap_line_outputs(raw)
    terms = criterion(outputs, line_targets)
    return sum(terms.values()) if terms else features[0].sum() * 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--init-pair", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=ROOT / "weights/checkpoint0119.pth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr-backbone", type=float, default=1e-6)
    parser.add_argument("--lr-encoder", type=float, default=2e-6)
    parser.add_argument("--lr-detector", type=float, default=1e-5)
    parser.add_argument("--lr-pair", type=float, default=1e-5)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--detection-weight", type=float, default=1.0)
    parser.add_argument("--line-weight", type=float, default=1.0)
    parser.add_argument("--lr-line", type=float, default=1e-5)
    parser.add_argument("--predicted-fraction", type=float, default=.5)
    parser.add_argument("--lowres-aug", type=float, default=.25)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--score-mode", choices=["max_type", "existence"], default="existence")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 <= args.predicted_fraction <= 1:
        parser.error("Invalid training settings")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output is not empty: {args.output}; select a new RUN_DIR")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    train_ds = JointHumanDataset(args.dataset, "train", lowres_probability=args.lowres_aug)
    val_ds = JointHumanDataset(args.dataset, "val", limit=args.val_limit)
    train_rows = read_json(args.dataset / "manifest.json")
    train_rows = [r for r in train_rows if r.get("split") == "train"]
    if any(r.get("source") != "webapp_human_fix" for r in train_rows):
        raise ValueError("Training manifest contains non-human source records")
    train_hashes = {r.get("source_hash") for r in train_rows}
    if len(train_ds) != len(train_rows) or not train_hashes:
        raise ValueError("Human-only training manifest mismatch")
    target_class_counts = {str(i): 0 for i in range(18)}
    target_bonds = 0
    for row in train_rows:
        truth = read_json(args.dataset / row["truth"])
        for atom in truth.get("atoms", []):
            target_class_counts[str(int(atom["category_id"]))] += 1
        for charge in truth.get("charges", []):
            label = str(charge.get("canonical_label", charge.get("label", ""))).strip()
            cid = CHARGE_ID.get(label)
            if cid is None:
                raise ValueError(f"Unsupported charge label {label!r}: {row['file_name']}")
            target_class_counts[str(cid)] += 1
        target_bonds += len(truth.get("bonds", []))
    charge_targets = sum(target_class_counts[str(i)] for i in range(13, 18))
    if not charge_targets:
        raise ValueError("Human Fix dataset has no explicit charge-box targets")

    pair_init = torch.load(args.init_pair, map_location="cpu", weights_only=False)
    if pair_init.get("architecture", "base") != "base":
        raise ValueError("This joint trainer currently expects the standard pair head")
    if pair_init.get("pair_classes") != PAIR_CLASSES:
        raise ValueError("Pair class schema mismatch")
    if pair_init["base_state_sha256"] != state_hash(load_model(args.base_checkpoint, "cpu").base):
        raise ValueError("Pair and base initialization checkpoint do not match")
    model = load_model(args.base_checkpoint, device, "base")
    model.pair_head.load_state_dict(pair_init["pair_head"], strict=True)
    # All components used by the deployed atom/bond path are trainable.
    model.base.backbone.requires_grad_(True)
    model.base.encoder.requires_grad_(True)
    model.base.box_decoder.requires_grad_(True)
    model.base.line_decoder.requires_grad_(True)
    model.pair_head.requires_grad_(True)

    cfg = YAMLConfig(str(ROOT / "config/train_mp123_full.yml"))
    criterion = cfg.criterion.box_criterion.to(device)
    line_criterion = cfg.criterion.line_criterion.to(device)
    optimizer = torch.optim.AdamW([
        {"params": list(model.base.backbone.parameters()), "lr": args.lr_backbone},
        {"params": list(model.base.encoder.parameters()), "lr": args.lr_encoder},
        {"params": list(model.base.box_decoder.parameters()), "lr": args.lr_detector},
        {"params": list(model.base.line_decoder.parameters()), "lr": args.lr_line},
        {"params": list(model.pair_head.parameters()), "lr": args.lr_pair},
    ], weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate)
    eval_args = argparse.Namespace(k=args.k, atom_cutoff=.3, pair_threshold=.3,
                                   score_mode=args.score_mode)

    initial_hashes = {name: state_hash(module) for name, module in
                      (("backbone", model.base.backbone), ("encoder", model.base.encoder),
                       ("box_decoder", model.base.box_decoder),
                       ("line_decoder", model.base.line_decoder), ("pair_head", model.pair_head))}
    history, best_score, best_epoch = [], float("-inf"), None
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        # Fine-tune weights, but do not estimate BN statistics from five source
        # molecules. Transformer dropout remains active in the decoders.
        model.base.backbone.eval(); model.base.encoder.eval()
        model.base.box_decoder.train()
        model.base.line_decoder.train()
        for module in list(model.base.box_decoder.modules()) + list(model.base.line_decoder.modules()):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        losses, detection_losses, pair_losses, line_losses = [], [], [], []
        positives = negatives = steps = zero_gradient_steps = pair_updates = 0
        for batch_index, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{**t, "centers": t["centers"].to(device), "labels": t["labels"].to(device),
                        "line_targets": {k: v.to(device) for k, v in t["line_targets"].items()}}
                       for t in targets]
            box_targets = [box_target_to_device(t, device) for t in targets]
            features_raw = model.base.encoder(model.base.backbone(images))
            features = features_raw[0] if isinstance(features_raw, tuple) else features_raw
            outputs = model.base.box_decoder(features, targets=box_targets)
            terms = criterion(outputs, box_targets)
            detection_loss = sum(terms.values())
            pair_loss, pos, neg = run_pair_loss(model, features, outputs, images, targets,
                                                args.k, args.predicted_fraction, args.score_mode)
            line_loss = run_line_loss(model, features, targets, line_criterion)
            loss = (args.detection_weight * detection_loss + args.pair_weight * pair_loss
                    + args.line_weight * line_loss)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite joint loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            named_groups = {"backbone": model.base.backbone, "encoder": model.base.encoder,
                            "box_decoder": model.base.box_decoder, "line_decoder": model.base.line_decoder,
                            "pair_head": model.pair_head}
            norms = {name: float(torch.nn.utils.clip_grad_norm_(
                [p for p in module.parameters() if p.requires_grad], 2.0,
                error_if_nonfinite=True)) for name, module in named_groups.items()}
            if norms["pair_head"] <= 1e-8:
                # A well-fitted pair batch can be saturated. Do not let AdamW
                # weight decay move its parameters without a meaningful signal.
                for parameter in model.pair_head.parameters():
                    parameter.grad = None
            missing = [name for name in ("backbone", "encoder", "box_decoder", "line_decoder")
                       if norms[name] <= 1e-8]
            if missing:
                raise RuntimeError("No gradient reached trainable groups: " + ",".join(missing))
            optimizer.step()
            pair_updates += int(norms["pair_head"] > 1e-8)
            losses.append(float(loss.detach())); detection_losses.append(float(detection_loss.detach()))
            pair_losses.append(float(pair_loss.detach())); line_losses.append(float(line_loss.detach()))
            positives += pos; negatives += neg; steps += 1
            if args.max_train_batches and steps >= args.max_train_batches:
                break
        if not steps:
            raise RuntimeError("No training batches")

        model.eval()
        bond_metrics = evaluate_pairs(model, val_loader, eval_args, device)
        entity_metrics = detect_metrics(model, val_loader, device)
        line_val = line_metrics(model, val_loader, device)
        predicted = bond_metrics["predicted_atoms"]["0.3"]
        scored_f1 = [line_val["f1"], predicted["f1"], entity_metrics["atom"]["f1"]]
        if entity_metrics["charge"]["truth"]:
            scored_f1.append(entity_metrics["charge"]["f1"])
        score = float(np.mean(scored_f1))
        row = {"epoch": epoch + 1, "steps": steps, "loss": float(np.mean(losses)),
               "detection_loss": float(np.mean(detection_losses)), "pair_loss": float(np.mean(pair_losses)),
               "line_loss": float(np.mean(line_losses)),
               "positive_pairs": positives, "negative_pairs": negatives, "last_batch_grad_norms": norms,
               "pair_head_optimizer_updates": pair_updates,
               "val_independent_line_decoder": line_val,
               "val_bond_predicted_atoms": predicted, "val_atom_charge_detection": entity_metrics,
               "selection_score": score}
        history.append(row)
        score_improved = score > best_score
        if score_improved:
            best_score, best_epoch = score, epoch + 1
        state = {"base": {k: v.detach().cpu().clone() for k, v in model.base.state_dict().items()},
                 "pair_head": {k: v.detach().cpu().clone() for k, v in model.pair_head.state_dict().items()},
                 "epoch": epoch + 1, "best_epoch": best_epoch, "history": history,
                 "base_sha256": state_hash(model.base), "pair_classes": PAIR_CLASSES,
                 "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 "selection_score": score}
        torch.save(state, args.output / "last.tmp.pth")
        (args.output / "last.tmp.pth").replace(args.output / "last.pth")
        if score_improved:
            torch.save(state, args.output / "best.pth")
        print(json.dumps(row), flush=True)
    final_hashes = {name: state_hash(module) for name, module in
                    (("backbone", model.base.backbone), ("encoder", model.base.encoder),
                     ("box_decoder", model.base.box_decoder),
                     ("line_decoder", model.base.line_decoder), ("pair_head", model.pair_head))}
    changed = {name: initial_hashes[name] != final_hashes[name] for name in initial_hashes}
    required_updates = {name: changed[name] for name in ("backbone", "encoder", "box_decoder", "line_decoder")}
    if not all(required_updates.values()):
        raise RuntimeError(f"Some required feature/detector weights did not change: {changed}")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    for tag in ("best", "last"):
        joint = torch.load(args.output / f"{tag}.pth", map_location="cpu", weights_only=False)
        base_path = args.output / f"{tag}_base.pth"
        torch.save(joint["base"], base_path)
        pair_artifact = {
            "pair_head": joint["pair_head"], "pair_classes": PAIR_CLASSES,
            "base_state_sha256": joint["base_sha256"],
            "base_checkpoint": str(base_path.resolve()), "architecture": "base",
            "selected_validation_threshold": .3,
            "args": {"k": args.k, "atom_cutoff": .3, "pair_threshold": .3,
                     "score_mode": args.score_mode},
        }
        torch.save(pair_artifact, args.output / f"{tag}_pair.pth")
    summary = {"epochs_completed": len(history), "best_epoch_by_fresh_holdout": best_epoch,
               "train_images": len(train_ds), "val_images": len(val_ds),
               "human_source_hashes": sorted(train_hashes), "trainable_groups": list(changed),
               "training_target_class_counts": target_class_counts,
               "training_charge_box_targets": charge_targets, "training_bond_edges": target_bonds,
               "weight_groups_changed": changed, "initial_weight_hashes": initial_hashes,
               "final_weight_hashes": final_hashes, "best_holdout_score": best_score,
               "elapsed_seconds": time.perf_counter() - start,
               "exported_inference_artifacts": ["best_base.pth", "best_pair.pth", "last_base.pth", "last_pair.pth"],
               "pair_loss_normalization": "Mean supervised pair loss within each image, then mean across images to prevent dense diagrams dominating.",
               "line_training": "Independent image-conditioned endpoint line decoder supervised directly by human bond geometry and bond type.",
               "note": "Full active detector/feature/line-decoder/pair fine-tune on human Fix data; BN running statistics remain frozen."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
