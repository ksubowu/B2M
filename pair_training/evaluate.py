"""Evaluate a trained pair branch, including the existing verified benchmark."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import PairDataset, collate
from .model import PAIR_CLASSES, state_hash
from .train import evaluate, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--base-checkpoint', type=Path)
    parser.add_argument('--group', help='Optional manifest group, e.g. source_pool')
    parser.add_argument('--cached-atom-run', type=Path,
                        help='Use cached assembled_graph predicted atoms for a controlled bond-head comparison')
    parser.add_argument('--validation-resolution', type=int, default=0,
                        help='Deterministic raster degradation for validation; -1 mixes native and 256px')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['pair_classes'] != PAIR_CLASSES:
        raise ValueError('Pair schema mismatch')
    model = load_model(args.base_checkpoint or checkpoint['base_checkpoint'], args.device,
                       checkpoint.get('architecture', 'base'))
    if state_hash(model.base) != checkpoint['base_state_sha256']:
        raise ValueError('Base model mismatch')
    model.pair_head.load_state_dict(checkpoint['pair_head'], strict=True)
    dataset = PairDataset(args.dataset, args.split, args.limit, allow_point_truth=True,
                          validation_resolution=args.validation_resolution)
    if args.group:
        dataset.records = [r for r in dataset.records if r.get('group') == args.group]
        if not dataset.records:
            raise ValueError('No images in requested group')
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)
    training_args = argparse.Namespace(**checkpoint['args'])
    training_args.pair_threshold = checkpoint['selected_validation_threshold']
    cases = []
    atom_provider = None
    if args.cached_atom_run:
        def atom_provider(target):
            path = args.cached_atom_run / 'cases' / Path(target['name']).with_suffix('.json')
            atoms = json.loads(path.read_text())['evidence']['assembled_graph']['atoms']
            centers = torch.tensor([a['center'] for a in atoms], device=args.device).reshape(-1, 2)
            centers = centers / torch.tensor(target['image_size'], device=args.device)
            labels = torch.tensor([a['label_id'] for a in atoms], device=args.device, dtype=torch.long)
            if ((labels < 0) | (labels > 12)).any():
                raise ValueError('Cached graph contains non-atom labels')
            return {'centers': centers, 'labels': labels}
    metrics = evaluate(model, loader, training_args, args.device, case_records=cases, atom_provider=atom_provider)
    report = {'checkpoint': str(args.checkpoint.resolve()), 'dataset': str(args.dataset.resolve()),
              'split': args.split, 'images': len(dataset),
              'checkpoint_validation_threshold': checkpoint['selected_validation_threshold'],
              'metrics': metrics, 'cases_at_validation_threshold': cases,
              'atom_source': str(args.cached_atom_run.resolve()) if args.cached_atom_run else 'frozen_atom_head_plus_nms',
              'group_filter': args.group}
    report['validation_resolution'] = args.validation_resolution
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
