"""Independent generated-image/graph/annotation integrity checks."""
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from rdkit import Chem

from .data import verify_split


def check(root):
    root = Path(root)
    split = verify_split(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    expected_atoms, expected_bonds = {}, {}
    total_atoms = total_bonds = 0
    for meta in manifest:
        truth = json.loads((root / meta['truth']).read_text())
        with Image.open(root / meta['image']) as image:
            image.load()
            assert list(image.size) == truth['image_size']
            width, height = image.size
        mol = Chem.MolFromMolBlock(truth['rendered_molfile'], removeHs=False)
        # Sanitization perceives aromaticity and turns depicted Kekule single/
        # double bonds into aromatic bonds. Check image labels against the raw
        # Molfile orders, and chemistry against the sanitized molecule.
        layout_mol = Chem.MolFromMolBlock(truth['rendered_molfile'], sanitize=False, removeHs=False)
        source = Chem.MolFromSmiles(truth['source_smiles'])
        assert mol is not None and source is not None
        assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(source), meta['file_name']
        assert mol.GetNumAtoms() == len(truth['atoms'])
        assert mol.GetNumBonds() == len(truth['bonds'])
        atoms = {a['index']: a for a in truth['atoms']}
        for atom in atoms.values():
            x, y, w, h = atom['bbox']
            cx, cy = atom['center']
            assert 0 <= x < x+w <= width and 0 <= y < y+h <= height
            assert x-1 <= cx <= x+w+1 and y-1 <= cy <= y+h+1
            assert mol.GetAtomWithIdx(atom['index']).GetSymbol() == atom['symbol']
        pairs = set()
        for bond in truth['bonds']:
            a, b = bond['begin'], bond['end']
            pair = tuple(sorted((a, b)))
            assert a != b and pair not in pairs
            pairs.add(pair)
            assert bond['line'] == atoms[a]['center'] + atoms[b]['center']
            native = layout_mol.GetBondBetweenAtoms(a, b)
            assert native is not None
            expected_category = {1.: 1, 2.: 2, 3.: 3, 1.5: 4}[native.GetBondTypeAsDouble()]
            assert bond['category_id'] == expected_category or (expected_category == 1 and bond['category_id'] in (5, 6))
        expected_atoms[meta['file_name']] = len(atoms) + sum(a['charge'] != 0 for a in atoms.values())
        expected_bonds[meta['file_name']] = len(pairs)
        total_atoms += len(atoms)
        total_bonds += len(pairs)
    for split_name in ['train', 'val']:
        for kind, expected in [('atom_charge', expected_atoms), ('bonds', expected_bonds)]:
            records = json.loads((root / f'annotations/{split_name}_{kind}.json').read_text())
            actual = Counter(r['file_name'] for r in records)
            for meta in manifest:
                if meta['split'] == split_name:
                    assert actual[meta['file_name']] == expected[meta['file_name']]
    return {'images_checked': len(manifest), 'atoms_checked': total_atoms,
            'bonds_checked': total_bonds, 'split_connectivities': split,
            'graph_image_annotation_integrity': 'passed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.dataset)
    (args.dataset / 'integrity_check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
