#!/usr/bin/env python3
"""Pair-branch inference on an image, without any ground-truth labels."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from pair_training.model import PAIR_CLASSES, decode_bond, state_hash
from pair_training.train import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON output path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--base-checkpoint', type=Path, help='Override relocated base checkpoint path')
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state['pair_classes'] != PAIR_CLASSES:
        raise ValueError('Unsupported pair label schema')
    model = load_model(args.base_checkpoint or state['base_checkpoint'], args.device,
                       state.get('architecture', 'base'))
    if state_hash(model.base) != state['base_state_sha256']:
        raise ValueError('Incorrect frozen base checkpoint')
    model.pair_head.load_state_dict(state['pair_head'], strict=True)
    model.eval()
    with Image.open(args.image) as source:
        image = source.convert('RGB')
    tensor = TF.normalize(TF.to_tensor(TF.resize(image, [640, 640])), [.485, .456, .406], [.229, .224, .225])
    with torch.no_grad():
        logits, atoms, pairs = model(tensor[None].to(args.device), k=state['args']['k'],
                                     atom_cutoff=state['args']['atom_cutoff'], native_sizes=[image.size])
        probs = logits[0].softmax(-1)
    threshold = args.threshold if args.threshold is not None else state['selected_validation_threshold']
    atom = atoms[0]
    mode = state['args'].get('score_mode')
    profile = 'single_head'
    if getattr(model.pair_head, 'is_resolution_adaptive', False):
        if model.pair_head.coarse(atom):
            mode, profile = 'existence', 'coarse'
        else:
            mode, profile = 'max_type', 'clear'
            threshold = float(model.pair_head.clear_threshold)
    coordinates = (atom['centers'].cpu() * torch.tensor(image.size)).tolist()
    records = [{'index': i, 'center': point, 'category_id': int(atom['labels'][i]),
                'score': float(atom['scores'][i])} for i, point in enumerate(coordinates)]
    bonds = []
    for pair, prob in zip(pairs[0].cpu().tolist(), probs.cpu()):
        score, label = prob[1:].max(0)
        label = int(label) + 1
        selection_score = 1-prob[0] if mode == 'existence' else score
        if float(selection_score) < threshold or selection_score <= prob[0]:
            continue
        category, reverse = decode_bond(label)
        a, b = pair[::-1] if reverse else pair
        bonds.append({'begin': a, 'end': b, 'category_id': category, 'score': float(selection_score),
                      'type_probability_given_bond': float(score / (1-prob[0]).clamp_min(1e-9)),
                      'none_score': float(prob[0]), 'line': coordinates[a] + coordinates[b]})
    output = {'image': str(args.image.resolve()), 'checkpoint': str(args.checkpoint.resolve()),
              'threshold': threshold, 'candidate_pairs': len(pairs[0]), 'atoms': records, 'bonds': bonds,
              'profile': profile,
              'note': 'Experimental pair branch; no OCR, chemistry repair, or production bond merging.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Output already exists; choose a new file')
    args.output.write_text(json.dumps(output, indent=2))
    overlay = image.convert('RGBA')
    layer = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(layer)
    for bond in bonds:
        draw.line(bond['line'], fill=(0, 100, 255, 80), width=7)
    for a in records:
        x, y = a['center']
        draw.ellipse([x-3, y-3, x+3, y+3], outline=(255, 0, 0, 255), width=1)
    Image.alpha_composite(overlay, layer).convert('RGB').save(args.output.with_suffix('.png'))
    print(json.dumps({'atoms': len(records), 'bonds': len(bonds), 'candidates': len(pairs[0]),
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
