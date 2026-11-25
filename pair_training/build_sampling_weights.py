"""Build training-only motif/stereo sampling weights without reading any test GT."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    records = [r for r in json.loads((args.dataset/'manifest.json').read_text()) if r['split'] == 'train']
    molecules = {r['id']: r['smiles'] for r in (json.loads(line) for line in (args.corpus/'molecules.jsonl').open())}
    sulfone = Chem.MolFromSmarts('[S](=[O])(=[O])')
    cached, counts, weights = {}, Counter(), []
    for record in records:
        key = record['molecule_id']
        if key not in cached:
            mol = Chem.MolFromSmiles(molecules[key])
            stereo = any(a.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                             Chem.ChiralType.CHI_TETRAHEDRAL_CCW) for a in mol.GetAtoms())
            flags = {'sulfone': mol.HasSubstructMatch(sulfone), 'tetrahedral_stereo': stereo,
                     'dense': mol.GetNumAtoms() >= 35}
            weight = max(4. if flags['sulfone'] else 1., 2. if stereo or flags['dense'] else 1.)
            cached[key] = weight, flags
        weight, flags = cached[key]
        weights.append(weight)
        for flag, present in flags.items():
            counts[flag] += int(present)
    result = {'manifest_sha256': hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest(),
              'training_images': len(records), 'training_flag_counts': dict(counts), 'weights': weights,
              'policy': 'Max of sulfone 4x, tetrahedral stereo 2x, dense>=35 atoms 2x, normal 1x; train split only.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'weights'}, indent=2))


if __name__ == '__main__':
    main()
