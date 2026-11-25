import json
import math
from pathlib import Path

import torch


CHARGE_NAME_TO_VALUE = {
    '-4': -4,
    '-2': -2,
    '-1': -1,
    '+1': 1,
    '+2': 2,
}

BOND_NAME_TO_GRAPH = {
    'single': {'order': 1, 'kind': 'single', 'stereo': None},
    '=': {'order': 2, 'kind': 'double', 'stereo': None},
    '#': {'order': 3, 'kind': 'triple', 'stereo': None},
    ':': {'order': 1.5, 'kind': 'aromatic', 'stereo': None},
    'wedge': {'order': 1, 'kind': 'single', 'stereo': 'wedge'},
    'dash': {'order': 1, 'kind': 'single', 'stereo': 'dash'},
    'other': {'order': 1, 'kind': 'single', 'stereo': None},
}


def load_chem_class_maps(classes_path):
    with open(classes_path, 'r') as f:
        data = json.load(f)
    atom_classes = {int(k): str(v) for k, v in data.get('atom_charge', {}).items()}
    bond_classes = {int(k): str(v) for k, v in data.get('bond', {}).items()}
    return atom_classes, bond_classes


def _to_cpu_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return list(x)


def _box_center(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5]


def _point_in_box(point, box, margin=0.0):
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def _point_distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(area_a + area_b - inter, 1e-6)
    return inter / union


def _line_length(line):
    return _point_distance(line[:2], line[2:4])


def _line_center(line):
    x1, y1, x2, y2 = line
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5]


def _best_atom_for_endpoint(endpoint, atoms):
    # Box containment must not override the nearest center. A large overlapping
    # label box can contain a neighboring vertex and steal its bond endpoint.
    best = min(atoms, key=lambda atom: _point_distance(endpoint, atom['center']))
    return best['index'], _point_distance(endpoint, best['center']), _point_in_box(endpoint,best['box'])


def _line_assignment_metrics(line, atom_a, atom_b):
    p1 = line[:2]
    p2 = line[2:4]
    center_err = _point_distance(p1, atom_a['center']) + _point_distance(p2, atom_b['center'])
    inside_count = int(_point_in_box(p1, atom_a['box'])) + int(_point_in_box(p2, atom_b['box']))
    return center_err, inside_count


def _select_atom_boxes(result, atom_classes, box_score_thresh):
    atoms = []
    charges = []
    boxes = _to_cpu_list(result['boxes'])
    labels = _to_cpu_list(result['labels'])
    scores = _to_cpu_list(result['scores'])

    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        class_name = atom_classes.get(int(label), str(label))
        if float(score) < box_score_thresh:
            continue
        item = {
            'raw_index': idx,
            'label_id': int(label),
            'label_name': class_name,
            'score': float(score),
            'box': [float(v) for v in box],
            'center': _box_center(box),
        }
        if class_name in CHARGE_NAME_TO_VALUE:
            item['charge_value'] = CHARGE_NAME_TO_VALUE[class_name]
            charges.append(item)
        else:
            atoms.append(item)

    atoms = sorted(atoms, key=lambda atom: (-atom['score'], atom['center'][0], atom['center'][1]))
    for new_idx, atom in enumerate(atoms):
        atom['index'] = new_idx
        atom['formal_charge'] = 0
        atom['charge_sources'] = []
    return atoms, charges


def _assign_charges(atoms, charges):
    if not atoms:
        return
    for charge in sorted(charges, key=lambda item: -item['score']):
        containing = [atom for atom in atoms if _point_in_box(charge['center'], atom['box'], margin=2.0)]
        if containing:
            target = max(containing, key=lambda atom: _box_iou(atom['box'], charge['box']))
        else:
            target = min(atoms, key=lambda atom: _point_distance(charge['center'], atom['center']))
        target['formal_charge'] += charge['charge_value']
        target['charge_sources'].append(charge)


def _select_line_candidates(result, atoms, bond_classes, line_score_thresh):
    candidates = []
    if len(atoms) < 2:
        return candidates

    lines = _to_cpu_list(result['lines'])
    labels = _to_cpu_list(result['line_labels'])
    scores = _to_cpu_list(result['line_scores'])

    atom_by_idx = {atom['index']: atom for atom in atoms}
    atom_by_raw = {atom['raw_index']:atom for atom in atoms}
    assigned = _to_cpu_list(result['line_atom_indices']) if 'line_atom_indices' in result else None
    if assigned is not None and len(assigned)!=len(lines):
        raise ValueError('line_atom_indices must have one ordered pair per line')

    for idx, (line, label, score) in enumerate(zip(lines, labels, scores)):
        if float(score) < line_score_thresh:
            continue
        line = [float(v) for v in line]
        label = int(label)
        bond_name = bond_classes.get(label, str(label))

        if assigned is not None:
            raw_a,raw_b=assigned[idx]
            if raw_a not in atom_by_raw or raw_b not in atom_by_raw:
                # A referenced atom was filtered; never silently reconnect it.
                continue
            a1=atom_by_raw[raw_a]['index'];a2=atom_by_raw[raw_b]['index']
            d1=_point_distance(line[:2],atom_by_idx[a1]['center']);d2=_point_distance(line[2:4],atom_by_idx[a2]['center'])
        else:
            a1, d1, _ = _best_atom_for_endpoint(line[:2], atoms)
            a2, d2, _ = _best_atom_for_endpoint(line[2:4], atoms)
        if a1 == a2:
            continue

        atom_a = atom_by_idx[a1]
        atom_b = atom_by_idx[a2]
        direct_err, direct_inside = _line_assignment_metrics(line, atom_a, atom_b)
        swap_err, swap_inside = _line_assignment_metrics(line, atom_b, atom_a)
        if assigned is None and swap_err < direct_err:
            begin_idx, end_idx = a2, a1
            center_err = swap_err
            inside_count = swap_inside
        else:
            begin_idx, end_idx = a1, a2
            center_err = direct_err
            inside_count = direct_inside

        begin_atom = atom_by_idx[begin_idx]
        end_atom = atom_by_idx[end_idx]
        candidates.append({
            'raw_index': idx,
            'label_id': label,
            'label_name': bond_name,
            'score': float(score),
            'line': line,
            'line_center': _line_center(line),
            'line_length': _line_length(line),
            'begin_atom_idx': begin_idx,
            'end_atom_idx': end_idx,
            'atom_pair': tuple(sorted((begin_idx, end_idx))),
            'center_error': center_err,
            'endpoint_distance_sum': d1 + d2,
            'inside_count': inside_count,
            'graph_bond': BOND_NAME_TO_GRAPH.get(bond_name, BOND_NAME_TO_GRAPH['single']),
        })
    return candidates


def _suppress_duplicate_lines(candidates):
    grouped = {}
    for cand in candidates:
        grouped.setdefault(cand['atom_pair'], []).append(cand)

    kept = []
    duplicates = []
    for atom_pair, items in grouped.items():
        best = sorted(
            items,
            key=lambda cand: (-cand['score'], cand['center_error'], -cand['inside_count'], cand['raw_index'])
        )[0]
        kept.append(best)
        for cand in items:
            if cand is not best:
                cand = dict(cand)
                cand['duplicate_of_raw_index'] = best['raw_index']
                duplicates.append(cand)
    kept.sort(key=lambda cand: (-cand['score'], cand['atom_pair']))
    return kept, duplicates


def _build_adjacency(atoms, bonds):
    atom_to_bonds = {atom['index']: [] for atom in atoms}
    for bond_idx, bond in enumerate(bonds):
        atom_to_bonds[bond['begin_atom_idx']].append(bond_idx)
        atom_to_bonds[bond['end_atom_idx']].append(bond_idx)
    return atom_to_bonds


def _atom_symbol(atom):
    name = atom['label_name']
    return '*' if name == 'other' else name


def assemble_molecular_graph(result, classes_path, box_score_thresh=0.5, line_score_thresh=0.35):
    atom_classes, bond_classes = load_chem_class_maps(classes_path)
    atoms, charges = _select_atom_boxes(result, atom_classes, box_score_thresh)
    _assign_charges(atoms, charges)
    line_candidates = _select_line_candidates(result, atoms, bond_classes, line_score_thresh)
    bonds, duplicate_lines = _suppress_duplicate_lines(line_candidates)
    atom_to_bonds = _build_adjacency(atoms, bonds)

    graph_atoms = []
    for atom in atoms:
        graph_atoms.append({
            'index': atom['index'],
            'symbol': _atom_symbol(atom),
            'label_name': atom['label_name'],
            'label_id': atom['label_id'],
            'formal_charge': atom['formal_charge'],
            'score': atom['score'],
            'box': atom['box'],
            'center': atom['center'],
            'bond_indices': atom_to_bonds[atom['index']],
            'charge_sources': [
                {
                    'label_name': item['label_name'],
                    'score': item['score'],
                    'box': item['box'],
                }
                for item in atom['charge_sources']
            ],
        })

    graph_bonds = []
    for bond_idx, bond in enumerate(bonds):
        graph_bonds.append({
            'index': bond_idx,
            'begin_atom_idx': bond['begin_atom_idx'],
            'end_atom_idx': bond['end_atom_idx'],
            'atom_pair': list(bond['atom_pair']),
            'label_name': bond['label_name'],
            'label_id': bond['label_id'],
            'score': bond['score'],
            'line': bond['line'],
            'center_error': bond['center_error'],
            'inside_count': bond['inside_count'],
            'graph_bond': bond['graph_bond'],
            'raw_index': bond['raw_index'],
        })

    graph = {
        'thresholds': {
            'box_score_thresh': box_score_thresh,
            'line_score_thresh': line_score_thresh,
        },
        'atoms': graph_atoms,
        'bonds': graph_bonds,
        'mappings': {
            'atom_to_bonds': atom_to_bonds,
            'duplicate_lines': duplicate_lines,
            'filtered_charge_boxes': charges,
        },
        'counts': {
            'kept_atoms': len(graph_atoms),
            'kept_bonds': len(graph_bonds),
            'duplicate_lines_removed': len(duplicate_lines),
            'charge_boxes_used': sum(len(atom['charge_sources']) for atom in graph_atoms),
        },
    }
    return graph


def result_from_graph(graph):
    return {
        'boxes': torch.tensor([atom['box'] for atom in graph['atoms']], dtype=torch.float32).reshape(-1, 4),
        'labels': torch.tensor([atom['label_id'] for atom in graph['atoms']], dtype=torch.int64),
        'scores': torch.tensor([atom['score'] for atom in graph['atoms']], dtype=torch.float32),
        'lines': torch.tensor([bond['line'] for bond in graph['bonds']], dtype=torch.float32).reshape(-1, 4),
        'line_labels': torch.tensor([bond['label_id'] for bond in graph['bonds']], dtype=torch.int64),
        'line_scores': torch.tensor([bond['score'] for bond in graph['bonds']], dtype=torch.float32),
    }


def maybe_build_rdkit(graph):
    try:
        from rdkit import Chem
    except Exception:
        return None

    mol = Chem.RWMol()
    atom_index_map = {}
    for atom in graph['atoms']:
        symbol = atom['symbol'] if atom['symbol'] != '*' else '*'
        rd_atom = Chem.Atom(symbol)
        rd_atom.SetFormalCharge(int(atom['formal_charge']))
        atom_index_map[atom['index']] = mol.AddAtom(rd_atom)

    for bond in graph['bonds']:
        begin = atom_index_map[bond['begin_atom_idx']]
        end = atom_index_map[bond['end_atom_idx']]
        kind = bond['graph_bond']['kind']
        if kind == 'double':
            bond_type = Chem.BondType.DOUBLE
        elif kind == 'triple':
            bond_type = Chem.BondType.TRIPLE
        elif kind == 'aromatic':
            bond_type = Chem.BondType.AROMATIC
        else:
            bond_type = Chem.BondType.SINGLE
        if mol.GetBondBetweenAtoms(begin, end) is None:
            mol.AddBond(begin, end, bond_type)

    mol = mol.GetMol()
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        smiles = None
    return {'smiles': smiles}


def default_classes_path_from_result_root(result_root):
    return str(Path(result_root).resolve() / 'annotations' / 'classes.json')
