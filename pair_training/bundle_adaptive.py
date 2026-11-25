"""Bundle clear/coarse specialists sharing identical frozen backbone/encoder/atoms."""
import argparse
import json
from pathlib import Path

import torch
from .model import ResolutionAdaptivePairHead, PAIR_CLASSES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clear-checkpoint', type=Path, required=True)
    p.add_argument('--coarse-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists')
    clear = torch.load(args.clear_checkpoint, map_location='cpu', weights_only=False)
    coarse = torch.load(args.coarse_checkpoint, map_location='cpu', weights_only=False)
    if clear['base_state_sha256'] != coarse['base_state_sha256']:
        raise ValueError('Specialists must share exactly the same frozen base')
    if clear['pair_classes'] != PAIR_CLASSES or coarse['pair_classes'] != PAIR_CLASSES:
        raise ValueError('Incompatible pair labels')
    if clear.get('architecture', 'base') != 'base' or coarse.get('architecture', 'base') != 'base':
        raise ValueError('This bundle expects two original-sized pair heads')
    if coarse['args'].get('score_mode') != 'existence':
        raise ValueError('Coarse specialist must be validated with existence scoring')
    head = ResolutionAdaptivePairHead()
    head.clear_head.load_state_dict(clear['pair_head'], strict=True)
    head.coarse_head.load_state_dict(coarse['pair_head'], strict=True)
    head.clear_threshold.fill_(clear['selected_validation_threshold'])
    # Below .5, the existing none-vs-exists check still requires existence>.5.
    # Record the effective threshold explicitly rather than an equivalent .3.
    threshold = max(.5, coarse['selected_validation_threshold'])
    config = dict(clear['args'])
    config.update(architecture='resolution_adaptive_v1', score_mode='existence')
    state = {'architecture': 'resolution_adaptive_v1', 'pair_head': head.state_dict(),
             'base_checkpoint': clear['base_checkpoint'], 'base_state_sha256': clear['base_state_sha256'],
             'pair_classes': PAIR_CLASSES, 'args': config, 'selected_validation_threshold': threshold,
             'profile_sources': {'clear': str(args.clear_checkpoint.resolve()),
                                 'coarse': str(args.coarse_checkpoint.resolve())},
             'policy': 'Native raster <320px uses coarse specialist and existence threshold; >=320px preserves clear specialist and its original max-type threshold.',
             'limitation': 'Native size is a quality proxy. Previously upscaled low-resolution images may require explicit effective-resolution metadata.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(json.dumps({k: v for k, v in state.items() if k not in ('pair_head', 'args')}, indent=2))


if __name__ == '__main__':
    main()
