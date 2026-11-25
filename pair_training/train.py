#!/usr/bin/env python3
"""Frozen epoch119 base with trainable pair branches and raster-domain adaptation."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'repo'), str(ROOT / 'repo/i2bl_webapp')]
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from engine.core import YAMLConfig
from i2unit_prediction import _load_checkpoint
from pair_training.data import (PairDataset, atom_mapping, collate, gt_atoms,
                                move_target, pair_targets, verify_split)
from pair_training.model import (FrozenEncoderPairModel, PAIR_CLASSES, knn_pairs,
                                 predicted_atoms, state_hash, PixelPairRefiner, AtomPairHead,
                                 ResolutionAdaptivePairHead, run_pair_head)


def load_model(base_checkpoint, device, architecture='base'):
    cfg = YAMLConfig(str(ROOT / 'config/train_mp123_full.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    base = cfg.model
    base.load_state_dict(_load_checkpoint(base_checkpoint), strict=True)
    if architecture not in ('base', 'pixel_refiner_v1', 'resolution_adaptive_v1'):
        raise ValueError('Unknown pair architecture: ' + architecture)
    head = (PixelPairRefiner() if architecture == 'pixel_refiner_v1' else
            ResolutionAdaptivePairHead() if architecture == 'resolution_adaptive_v1' else None)
    model=FrozenEncoderPairModel(base, head=head).to(device)
    payload=torch.load(base_checkpoint,map_location='cpu',weights_only=False)
    if isinstance(payload,dict) and payload.get('text_entity_route'):
        from pair_training.text_decoder_route import load_reference_heads
        load_reference_heads(model,payload,device)
    return model


def metrics_row(counts):
    result = dict(counts)
    tp, fp, fn = result['tp'], result['fp'], result['fn']
    result.update(precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
                  f1=2 * tp / max(1, 2 * tp + fp + fn),
                  exact_atom_bond_rate=result['exact_images'] / max(1, result['images']))
    return result


def pair_optimizer_step(loss, optimizer, parameters):
    """A saturated, correctly classified batch can have exactly zero gradient."""
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite pair loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not any(p.grad is not None for p in parameters):
        raise RuntimeError('Pair loss is disconnected from trainable parameters')
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    # Avoid weight decay/momentum updates on a fully saturated batch, but count
    # it as a valid supervised batch and continue with the rest of the epoch.
    if float(norm) > 0:
        optimizer.step()
    return float(norm)


@torch.no_grad()
def evaluate(model, loader, args, device, case_records=None, atom_provider=None):
    model.eval()
    thresholds = sorted(set([.3, .5, .7, .9, args.pair_threshold]))
    stats = {mode: {str(t): {'tp': 0, 'fp': 0, 'fn': 0, 'candidate_gt_bonds': 0,
                            'gt_bonds': 0, 'images': 0, 'exact_images': 0,
                            'atom_tp': 0, 'atom_fp': 0, 'atom_fn': 0}
                    for t in thresholds} for mode in ['gt_atoms', 'predicted_atoms']}
    for images, targets in loader:
        images = images.to(device)
        targets = [move_target(t, device) for t in targets]
        features, outputs = model.extract(images)
        for mode in stats:
            if mode == 'gt_atoms':
                atoms = [gt_atoms(t) for t in targets]
            elif atom_provider is not None:
                atoms = [atom_provider(t) for t in targets]
            else:
                atoms = [predicted_atoms(outputs, i, args.atom_cutoff) for i in range(len(targets))]
            for atom, target in zip(atoms, targets):
                atom['raster_resolution'] = target.get('raster_resolution', min(target.get('image_size', [640])))
            pairs = [knn_pairs(a['centers'], args.k) for a in atoms]
            logits = run_pair_head(model.pair_head, features, atoms, pairs, images)
            for atom, pair, logit, target in zip(atoms, pairs, logits, targets):
                expected, mapping = pair_targets(atom, pair, target, ignore_unmatched=False)
                probs = logit.softmax(-1)
                bond_prob, bond_type = probs[:, 1:].max(-1)
                bond_type += 1
                if getattr(args, 'score_mode', 'max_type') == 'existence':
                    bond_prob = 1-probs[:, 0]
                atom_tp = sum(int(atom['labels'][i]) == int(target['labels'][j]) for i, j in mapping.items())
                atom_fp, atom_fn = len(atom['centers']) - atom_tp, len(target['centers']) - atom_tp
                for threshold in thresholds:
                    row = stats[mode][str(threshold)]
                    active_prob, active_threshold = bond_prob, threshold
                    profile = 'single_head'
                    if getattr(model.pair_head, 'is_resolution_adaptive', False):
                        if model.pair_head.coarse(atom):
                            active_prob, profile = 1-probs[:, 0], 'coarse'
                        else:
                            active_prob = probs[:, 1:].max(-1).values
                            active_threshold, profile = float(model.pair_head.clear_threshold), 'clear'
                    selected = (active_prob >= active_threshold) & (active_prob > probs[:, 0])
                    tp = int((selected & (bond_type == expected) & (expected > 0)).sum())
                    fp = int(selected.sum()) - tp
                    fn = len(target['edges']) - tp
                    for key, value in [('tp', tp), ('fp', fp), ('fn', fn),
                                       ('candidate_gt_bonds', int((expected > 0).sum())),
                                       ('gt_bonds', len(target['edges'])), ('images', 1),
                                       ('exact_images', int(fp == fn == atom_fp == atom_fn == 0)),
                                       ('atom_tp', atom_tp), ('atom_fp', atom_fp), ('atom_fn', atom_fn)]:
                        row[key] += value
                    if case_records is not None and threshold == args.pair_threshold:
                        # Previous benchmark treated wedge/dash endpoints as
                        # unordered and checked charge separately. Export an
                        # explicitly named compatible atom/bond-only metric.
                        category_map = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6, 5, 6], device=expected.device)
                        compatible_tp = int((selected & (expected > 0) &
                                             (category_map[bond_type] == category_map[expected])).sum())
                        compatible_fp = int(selected.sum()) - compatible_tp
                        compatible_fn = len(target['edges']) - compatible_tp
                        case_records.append({'image': target['name'], 'group': target.get('group'),
                                             'profile': profile, 'decision_threshold': active_threshold,
                                             'mode': mode, 'threshold': threshold, 'tp': tp, 'fp': fp, 'fn': fn,
                                             'gt_bonds': len(target['edges']), 'candidates': len(pair),
                                             'candidate_gt_bonds': int((expected > 0).sum()),
                                             'atom_tp': atom_tp, 'atom_fp': atom_fp, 'atom_fn': atom_fn,
                                             'exact_atom_bond_graph': fp == fn == atom_fp == atom_fn == 0,
                                             'undirected_typed_tp': compatible_tp,
                                             'undirected_typed_fp': compatible_fp,
                                             'undirected_typed_fn': compatible_fn,
                                             'extra_connections': int((selected & (expected == 0)).sum()),
                                             'missing_connections': len(target['edges']) - int((selected & (expected > 0)).sum()),
                                             'wrong_bond_types': int((selected & (expected > 0)).sum()) - compatible_tp,
                                             'wrong_wedge_directions': compatible_tp - tp,
                                             'exact_undirected_atom_bond_graph': compatible_fp == compatible_fn == atom_fp == atom_fn == 0})
    return {mode: {t: metrics_row(counts) for t, counts in rows.items()} for mode, rows in stats.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, default=ROOT / 'weights/checkpoint0119.pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--predicted-fraction', type=float, default=.5)
    parser.add_argument('--jitter', type=float, default=.002)
    parser.add_argument('--atom-cutoff', type=float, default=.4)
    parser.add_argument('--pair-threshold', type=float, default=.5)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--max-train-batches', type=int, default=0)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--architecture', choices=['base', 'pixel_refiner_v1'], default='base')
    parser.add_argument('--score-mode', choices=['max_type', 'existence'], default='max_type')
    parser.add_argument('--lowres-aug', type=float, default=0.)
    parser.add_argument('--validation-resolution', type=int, default=0)
    parser.add_argument('--distill-normal', type=float, default=0.)
    parser.add_argument('--selection-metric', choices=['bond_f1', 'graph_exact'], default='bond_f1')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--sample-weights', type=Path)
    args = parser.parse_args()
    if not 0 <= args.predicted_fraction <= 1 or args.epochs < 1 or args.batch_size < 1 or args.k < 1:
        parser.error('Invalid training options')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Output is not empty; use a new directory or --resume checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    split = verify_split(args.dataset)
    training = PairDataset(args.dataset, 'train', args.train_limit, lowres_probability=args.lowres_aug)
    validation = PairDataset(args.dataset, 'val', args.val_limit, validation_resolution=args.validation_resolution)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    sampling_hash = None
    if args.sample_weights:
        sampling_hash = hashlib.sha256(args.sample_weights.read_bytes()).hexdigest()
        weights = json.loads(args.sample_weights.read_text())
        current_hash = hashlib.sha256((args.dataset / 'manifest.json').read_bytes()).hexdigest()
        if weights['manifest_sha256'] != current_hash:
            raise ValueError('Sampling weights belong to a different dataset')
        values = weights['weights'][:len(training)]
        if len(values) != len(training) or min(values) <= 0:
            raise ValueError('Invalid sampling weights')
        sampler = WeightedRandomSampler(values, len(training), replacement=True, generator=generator)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, generator=generator,
                        num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
    model = load_model(args.base_checkpoint, device, args.architecture)
    frozen_before = state_hash(model.base)
    if args.init_checkpoint:
        if args.resume:
            parser.error('--init-checkpoint and --resume are mutually exclusive')
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if initial['base_state_sha256'] != frozen_before:
            raise ValueError('Initialization base checkpoint mismatch')
        if args.architecture == 'pixel_refiner_v1' and initial.get('architecture', 'base') == 'base':
            model.pair_head.old_head.load_state_dict(initial['pair_head'], strict=True)
        elif args.architecture == initial.get('architecture', 'base'):
            model.pair_head.load_state_dict(initial['pair_head'], strict=True)
        else:
            raise ValueError('Incompatible initialization architecture')
    frozen_prior_before = state_hash(model.pair_head.old_head) if args.architecture == 'pixel_refiner_v1' else None
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    manifest_hash = hashlib.sha256((args.dataset / 'manifest.json').read_bytes()).hexdigest()
    start_epoch, best_f1, history = 0, -1., []
    best_selection = (-1., -1.)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state['base_state_sha256'] != frozen_before or state['manifest_sha256'] != manifest_hash:
            raise ValueError('Resume base weights or dataset mismatch')
        if state['pair_classes'] != PAIR_CLASSES:
            raise ValueError('Pair label schema mismatch')
        if state.get('architecture', 'base') != args.architecture:
            raise ValueError('Resume architecture mismatch')
        if state.get('sample_weights_sha256') != sampling_hash:
            raise ValueError('Resume sampling policy mismatch')
        for key in ['k', 'atom_cutoff', 'predicted_fraction', 'jitter', 'batch_size', 'train_limit', 'val_limit']:
            if state['args'][key] != getattr(args, key):
                raise ValueError('Resume option mismatch: ' + key)
        for key, default in [('score_mode', 'max_type'), ('lowres_aug', 0.), ('validation_resolution', 0),
                             ('distill_normal', 0.), ('selection_metric', 'bond_f1')]:
            if state['args'].get(key, default) != getattr(args, key):
                raise ValueError('Resume option mismatch: ' + key)
        model.pair_head.load_state_dict(state['pair_head'], strict=True)
        if args.architecture == 'pixel_refiner_v1':
            frozen_prior_before = state_hash(model.pair_head.old_head)
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, best_f1, history = state['epoch'], state['best_f1'], state['history']
        best_selection = tuple(state.get('best_selection', (best_f1, -1.)))
        torch.set_rng_state(state['torch_rng'].cpu())
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        generator.set_state(state['loader_rng'].cpu())
        if device.type == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
    if start_epoch >= args.epochs:
        parser.error('--epochs is the total epoch count and must exceed resumed epoch')
    teacher = None
    if args.distill_normal:
        teacher_path = args.init_checkpoint or (state['args'].get('init_checkpoint') if args.resume else None)
        if not teacher_path:
            parser.error('Normal-image distillation requires an initialization checkpoint')
        teacher_state = torch.load(teacher_path, map_location=device, weights_only=False)
        if teacher_state.get('architecture', 'base') != 'base':
            parser.error('The normal-image teacher must be an original pair-head checkpoint')
        teacher = AtomPairHead().to(device).requires_grad_(False).eval()
        teacher.load_state_dict(teacher_state['pair_head'], strict=True)
    start = time.perf_counter()
    reloaded = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses, steps, positive, negative, ignored = [], 0, 0, 0, 0
        zero_gradient_batches = 0
        for images, targets in loader:
            images = images.to(device)
            targets = [move_target(t, device) for t in targets]
            features, atom_outputs = model.extract(images)
            atoms = [predicted_atoms(atom_outputs, i, args.atom_cutoff)
                     if random.random() < args.predicted_fraction else gt_atoms(t, args.jitter, shuffle=True)
                     for i, t in enumerate(targets)]
            pairs = [knn_pairs(a['centers'], args.k) for a in atoms]
            logits = run_pair_head(model.pair_head, features, atoms, pairs, images)
            labels = [pair_targets(a, p, t)[0] for a, p, t in zip(atoms, pairs, targets)]
            all_labels, all_logits = torch.cat(labels), torch.cat(logits)
            valid = all_labels != -100
            if not valid.any():
                continue
            if args.score_mode == 'existence':
                supervised = all_logits[valid]
                supervised_labels = all_labels[valid]
                positive_mask = supervised_labels > 0
                existence_logits = torch.logsumexp(supervised[:, 1:], -1) - supervised[:, 0]
                loss = F.binary_cross_entropy_with_logits(existence_logits, positive_mask.float())
                if positive_mask.any():
                    loss = loss + F.cross_entropy(supervised[positive_mask, 1:], supervised_labels[positive_mask]-1)
            else:
                loss = F.cross_entropy(all_logits[valid], all_labels[valid])
            if teacher is not None:
                normal_mask = torch.cat([torch.full_like(label, not t.get('raster_augmented', False), dtype=torch.bool)
                                         for label, t in zip(labels, targets)]) & valid
                if normal_mask.any():
                    with torch.no_grad():
                        teacher_logits = torch.cat(teacher(features, atoms, pairs))
                    loss = loss + args.distill_normal * 4 * F.kl_div(
                        F.log_softmax(all_logits[normal_mask]/2, -1),
                        F.softmax(teacher_logits[normal_mask]/2, -1), reduction='batchmean')
            norm = pair_optimizer_step(loss, optimizer, trainable)
            zero_gradient_batches += int(norm == 0)
            losses.append(float(loss.detach()))
            steps += 1
            positive += int((all_labels > 0).sum())
            negative += int((all_labels == 0).sum())
            ignored += int((all_labels == -100).sum())
            if args.log_interval and steps % args.log_interval == 0:
                print(json.dumps({'epoch': epoch+1, 'train_step': steps, 'batch_loss': losses[-1]}), flush=True)
            if args.max_train_batches and steps >= args.max_train_batches:
                break
        if not steps:
            raise RuntimeError('No supervised candidate pairs in epoch')
        validation_metrics = evaluate(model, val_loader, args, device)
        row = {'epoch': epoch + 1, 'steps': steps, 'loss': sum(losses) / len(losses),
               'positive_pairs': positive, 'negative_pairs': negative, 'ignored_pairs': ignored,
               'zero_gradient_batches': zero_gradient_batches,
               'validation': validation_metrics}
        history.append(row)
        def selection_key(row):
            return ((row['exact_atom_bond_rate'], row['f1']) if args.selection_metric == 'graph_exact'
                    else (row['f1'], row['exact_atom_bond_rate']))
        best_threshold, selected = max(validation_metrics['predicted_atoms'].items(), key=lambda item: selection_key(item[1]))
        improved = selection_key(selected) > best_selection
        best_selection = max(best_selection, selection_key(selected))
        best_f1 = max(best_f1, selected['f1'])
        state = {'pair_head': model.pair_head.state_dict(), 'optimizer': optimizer.state_dict(),
                 'epoch': epoch + 1, 'best_f1': best_f1, 'history': history,
                 'best_selection': best_selection,
                 'pair_classes': PAIR_CLASSES, 'base_checkpoint': str(args.base_checkpoint.resolve()),
                 'architecture': args.architecture,
                 'sample_weights_sha256': sampling_hash,
                 'base_state_sha256': frozen_before, 'manifest_sha256': manifest_hash,
                 'selected_validation_threshold': float(best_threshold),
                 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
                 'numpy_rng': np.random.get_state(), 'loader_rng': generator.get_state(),
                 'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
        checkpoint = output / 'last.pth'
        temporary = output / 'last.tmp.pth'
        torch.save(state, temporary)
        temporary.replace(checkpoint)
        if improved:
            torch.save(state, output / 'best.pth')
        # Verify serialization numerically on actual predicted-atom validation data.
        if not reloaded:
            model.eval()
            probe = torch.stack([validation[i][0] for i in range(min(args.batch_size, len(validation)))])
            with torch.no_grad():
                before = model(probe.to(device), k=args.k, atom_cutoff=args.atom_cutoff)[0]
                saved = torch.load(checkpoint, map_location=device, weights_only=False)
                model.pair_head.load_state_dict(saved['pair_head'], strict=True)
                after = model(probe.to(device), k=args.k, atom_cutoff=args.atom_cutoff)[0]
            for a, b in zip(before, after):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            reloaded = True
        print(json.dumps({'epoch': epoch + 1, 'steps': steps, 'loss': row['loss'],
                          'positive_pairs': positive, 'negative_pairs': negative,
                          'zero_gradient_batches': zero_gradient_batches,
                          'best_val_threshold': best_threshold, 'predicted_atom_validation': selected}), flush=True)
        (output / 'history.json').write_text(json.dumps(history, indent=2))
    frozen_after = state_hash(model.base)
    if frozen_after != frozen_before:
        raise RuntimeError('Frozen base parameters or buffers changed')
    if frozen_prior_before is not None and state_hash(model.pair_head.old_head) != frozen_prior_before:
        raise RuntimeError('Frozen prior pair head changed')
    summary = {'epochs_completed': args.epochs, 'dataset': str(args.dataset.resolve()),
               'train_images': len(training), 'val_images': len(validation), 'split_connectivities': split,
               'trainable_parameters': sum(p.numel() for p in trainable),
               'architecture': args.architecture,
               'weighted_sampling': args.sample_weights is not None,
               'frozen_prior_unchanged': frozen_prior_before is not None,
               'base_state_sha256': frozen_before, 'frozen_base_unchanged': True,
               'checkpoint_reload_identical': reloaded, 'best_validation_f1': best_f1,
               'elapsed_seconds': time.perf_counter() - start,
               'peak_cuda_memory_mb': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
               'note': 'Frozen-base pair training. Metrics include missing candidate bonds. Exact graph excludes charge/isotope; wedge directions are checked. Validation-selected thresholds need independent evaluation.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
