"""Best-validation pair/anchor/text inference bundle for the WebApp."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from torchvision.ops import box_convert

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_RUNTIME = None


def _load_runtime(bundle_path, device_name=None):
    global _RUNTIME
    bundle = Path(bundle_path).resolve()
    device = torch.device(device_name or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    key = (str(bundle), str(device))
    if _RUNTIME is not None and _RUNTIME['key'] == key:
        return _RUNTIME
    from pair_training.train import load_model
    from pair_training.model import state_hash
    from pair_training.pretrained_ocr import PretrainedEasyOCR
    from pair_training.text_sequence import NativeROITextHead

    pair_ck = torch.load(bundle / 'pair.pth', map_location='cpu', weights_only=False)
    model = load_model(bundle / 'base.pth', device, pair_ck['architecture']).eval()
    if state_hash(model.base) != pair_ck['base_state_sha256']:
        raise ValueError('Pair bundle base checkpoint hash mismatch')
    model.pair_head.load_state_dict(pair_ck['pair_head'], strict=True)
    model.pair_head.eval().requires_grad_(False)

    anchor_ck = torch.load(bundle / 'anchor.pth', map_location='cpu', weights_only=False)
    if anchor_ck['base_hash'] != pair_ck['base_state_sha256']:
        raise ValueError('Anchor checkpoint was trained for a different base model')
    anchor = NativeROITextHead(**anchor_ck['config']).to(device).eval().requires_grad_(False)
    anchor.load_state_dict(anchor_ck['state_dict'], strict=True)
    ocr = PretrainedEasyOCR(model_dir=PROJECT_ROOT / 'weights' / 'easyocr', device=str(device),
                            checkpoint=bundle / 'ocr.pth').eval()
    classes = PROJECT_ROOT / 'dataset_pair_real_20261015' / 'annotations' / 'classes.json'
    if not classes.exists():
        raise FileNotFoundError(f'Pair model class map not found: {classes}')
    _RUNTIME = dict(key=key, bundle=bundle, device=device, model=model, pair_ck=pair_ck,
                    anchor=anchor, ocr=ocr, classes=classes)
    return _RUNTIME


def _assemble(atom, bonds, image, classes_path):
    from engine.rtv4.chem_graph import assemble_molecular_graph
    boxes = box_convert(atom['boxes'].detach().cpu(), 'cxcywh', 'xyxy')
    boxes *= torch.tensor([*image.size, *image.size], dtype=boxes.dtype)
    result = dict(boxes=boxes, labels=atom['labels'].detach().cpu(),
                  scores=atom['scores'].detach().cpu(),
                  lines=torch.tensor([b['line'] for b in bonds], dtype=torch.float32).reshape(-1, 4),
                  line_labels=torch.tensor([b['category_id'] for b in bonds], dtype=torch.long),
                  line_scores=torch.tensor([b['score'] for b in bonds], dtype=torch.float32),
                  line_atom_indices=torch.tensor([[b['begin'], b['end']] for b in bonds], dtype=torch.long).reshape(-1, 2))
    return assemble_molecular_graph(result, classes_path=str(classes_path), box_score_thresh=0., line_score_thresh=0.)


def _infer_graph(atom, image, features, runtime, ck):
    from pair_training.model import knn_pairs, decode_bond, pair_decision_policy
    from pair_training.text_atoms import decode_reference
    from engine.rtv4.chem_graph import assemble_molecular_graph

    model = runtime['model']
    with torch.no_grad():
        pairs = knn_pairs(atom['centers'], ck['args']['k'])
        logits = model.pair_head(features, [atom], [pairs])[0]
    threshold, score_mode, _profile = pair_decision_policy(model.pair_head, atom, ck)
    decoded = decode_reference(atom, pairs, logits, image.size, threshold, score_mode)
    bonds = decoded['bonds']
    graph = _assemble(atom, bonds, image, runtime['classes'])

    if ck.get('component_bridge_repair'):
        from pair_training.component_bridges import bridge_candidates, POLICY
        if ck['component_bridge_repair'] != POLICY:
            raise ValueError('Unsupported component bridge policy in checkpoint')
        centers = (atom['centers'].detach().cpu() * torch.tensor(image.size)).numpy()
        additions, audit = bridge_candidates(graph, image,
            box_convert(atom['boxes'].detach().cpu(), 'cxcywh', 'xyxy').mul(
                torch.tensor([*image.size, *image.size])).numpy(),
            centers, pairs.cpu().numpy(), logits.softmax(-1).detach().cpu().numpy(),
            'existence' if coarse else 'max_type')
        if additions:
            graph = _assemble(atom, bonds + additions, image, runtime['classes'])
        graph['component_bridge_repair'] = audit

    if ck.get('text_entity_route'):
        from pair_training.text_decoder_route import POLICY, graph_quality, choose_route
        if ck['text_entity_route'] != POLICY or not hasattr(model, 'original_box_decoder'):
            raise ValueError('Text decoder route metadata/model mismatch')
        from pair_training.model import predicted_atoms
        from pair_training.text_atoms import apply_text_anchors
        with torch.no_grad():
            old_outputs = model.original_box_decoder(features)
            original = predicted_atoms(old_outputs, 0, cutoff=POLICY['atom_cutoff'])
            original['raster_resolution'] = min(image.size)
            original, _ = apply_text_anchors(original, image, features, runtime['anchor'])
            old_graph = _infer_graph(original, image, features, runtime,
                                     {**ck, 'text_entity_route': None})
        reason = choose_route(graph_quality(graph, image), graph_quality(old_graph, image))
        if reason != 'learner':
            graph = old_graph
        graph['text_entity_route'] = dict(policy=POLICY, selected=reason)
    return graph


def predict_pair_bundle(image_path, bundle_path, device_name=None):
    """Return the selected learned-entity/anchor/pair/adapted-OCR prediction."""
    from pair_training.model import predicted_atoms
    from pair_training.text_atoms import detect_label_regions, reconcile_text_atoms, apply_text_anchors
    from pair_training.notebook_ocr import recognize_multiscale, labeled_cxsmiles, expand_labels
    from pair_training.chemistry import build_rdkit_molecule

    runtime = _load_runtime(bundle_path, device_name)
    image = Image.open(image_path).convert('RGB')
    tensor = TF.normalize(TF.to_tensor(TF.resize(image, [640, 640])),
                          [.485, .456, .406], [.229, .224, .225])[None].to(runtime['device'])
    with torch.no_grad():
        features, outputs = runtime['model'].extract(tensor)
        atom = predicted_atoms(outputs, 0, cutoff=runtime['pair_ck']['args']['atom_cutoff'])
        atom['raster_resolution'] = min(image.size)
    regions = detect_label_regions(image, scales=(1, 2), min_confidence=.85)
    atom, region_audit = reconcile_text_atoms(atom, image, regions)
    atom['raster_resolution'] = min(image.size)
    atom, anchor_audit = apply_text_anchors(atom, image, features, runtime['anchor'])
    graph = _infer_graph(atom, image, features, runtime, runtime['pair_ck'])
    labels, candidates = recognize_multiscale(graph, image, scales=(1, 2, 3, 4),
                                               margins=(2, 4), min_confidence=.5,
                                               recognizer=runtime['ocr'].recognize_crop)
    from pair_training.hydrogen_notation import collapse_hydrogen_notation
    graph, labels, hydrogen_audit = collapse_hydrogen_notation(graph, labels)
    for atom in graph['atoms']:
        notation = atom.get('atom_notation')
        if notation and atom['index'] not in labels:
            labels[atom['index']] = dict(text=notation, raw_text=notation, score=1.0,
                                         eligible_for_expansion=False, smiles='[*]',
                                         merged_atom_indices=[atom['index']])
    mol, chemistry = build_rdkit_molecule(graph, infer_double_stereo=False)
    _, cx = labeled_cxsmiles(mol, graph, labels)
    expanded_mol, expanded = expand_labels(mol, graph, labels)
    return dict(image=image, graph=graph, labels=labels, candidates=candidates,
                region_audit=region_audit, anchor_audit=anchor_audit,
                cxsmiles=cx.get('cxsmiles'), expanded_smiles=expanded.get('smiles'),
                chemistry=chemistry, expanded=expanded,
                hydrogen_notation=hydrogen_audit,
                backend='dual_decoder_pilot_20261022', bundle=str(runtime['bundle']),
                text_detector='PaddleOCR multiscale regions', text_recognizer=runtime['ocr'].backend)
