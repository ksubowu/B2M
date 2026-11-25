#!/usr/bin/env python3
"""I2BL + Ketcher WebApp — FastAPI-based chemical structure recognition with TODO2."""
import hashlib, math
import json, os, re, secrets, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image, UnidentifiedImageError
from rdkit import Chem
from rdkit.Chem import AllChem
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

RTDETR_ROOT = Path(os.environ.get("I2BL_RTDETR_ROOT", Path(__file__).resolve().parents[1])).resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RTDETR_ROOT))
sys.path.insert(0, str(RTDETR_ROOT / "i2bl_webapp"))
sys.path.insert(0, str(PROJECT_ROOT))

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "webapp_static"
RESULTS_DATA_DIR = Path(os.environ.get("I2BL_RESULTS_DATA_DIR", BASE_DIR / "results_data"))
FIXED_DATA_DIR = Path(os.environ.get("I2BL_FIXED_DATA_DIR", BASE_DIR / "fixed_data"))
FIXED_RECORDS_DIR = FIXED_DATA_DIR / "records"
TEST_DIR = Path(os.environ.get("I2BL_EXAMPLES_DIR", BASE_DIR / "test")).resolve()
PASTED_EXAMPLES_DIR = Path(os.environ.get("I2BL_PASTED_EXAMPLES_DIR", BASE_DIR / "pasted_examples")).resolve()
TRAINING_PAIRS_DIR = Path(os.environ.get("I2BL_TRAINING_PAIRS_DIR", FIXED_DATA_DIR / "training_pairs")).resolve()
KETCHER_CANDIDATES = [
    BASE_DIR / "ketcher-master" / "demo" / "build",
    BASE_DIR / "ketcher-master" / "example" / "dist" / "standalone",
    BASE_DIR / "ketcher",
]

# Default model config
DEFAULT_CONFIG = Path(os.environ.get("I2BL_CONFIG", RTDETR_ROOT.parent / "config" / "train_mp123_full.yml"))
DEFAULT_RESUME = Path(os.environ.get("I2BL_CHECKPOINT", PROJECT_ROOT / "runs/human_joint_full_20261027/best_base.pth"))
DEFAULT_CLASSES = Path(os.environ.get("I2BL_CLASSES", PROJECT_ROOT / "runs/pair_human_only_all_20261026/annotations/classes.json"))
DEFAULT_ATOM_CUTOFF = 0.5
DEFAULT_LINE_CUTOFF = 0.4
DEFAULT_ATOM_NMS = 0.55
DEFAULT_CHARGE_NMS = 0.55
DEFAULT_INFERENCE_PROFILE = os.environ.get("I2BL_INFERENCE_PROFILE", "legacy").strip().lower()
DEFAULT_PAIR_BUNDLE = Path(os.environ.get(
    "I2BL_PAIR_BUNDLE",
    PROJECT_ROOT / "runs" / "chemical_structure_human_pair_update_20261027" / "bundle",
)).resolve()
PAIR_DEVICE = os.environ.get("I2BL_PAIR_DEVICE")

# Global model (lazy loaded)
_model = None
_postprocessor = None
_cfg = None
_box_classes = None
_line_classes = None
_image_size = None
_mean = None
_std = None
_classes_path = None
_device = None
_inference_dependencies_loaded = False


def _load_inference_dependencies():
    """Delay CUDA/PyTorch/Paddle imports until the first prediction request."""
    global _inference_dependencies_loaded, torch
    global assemble_molecular_graph, expand_molecule_graph, recognize_rgroup_labels
    global _get_paddleocr_instances, _paddleocr_recognize
    global _build_model, _load_classes, _infer_runtime_from_config
    global _infer_single, _filter_sort_and_nms_boxes, _dedupe_lines_by_nearest_boxes
    global _build_structured_raw_result
    if _inference_dependencies_loaded:
        return

    import torch as torch_module
    from engine.rtv4.chem_graph import assemble_molecular_graph as assemble_graph
    from engine.rtv4.chem_ocr_expand import (
        expand_molecule_graph as expand_graph,
        recognize_rgroup_labels as recognize_labels,
        _get_paddleocr_instances as get_paddleocr_instances,
        _paddleocr_recognize as paddleocr_recognize,
    )
    from i2unit_prediction import (
        _build_model as build_model,
        _load_classes as load_classes,
        _infer_runtime_from_config as infer_runtime,
        _infer_single as infer_single,
        _filter_sort_and_nms_boxes as filter_boxes,
        _dedupe_lines_by_nearest_boxes as dedupe_lines,
        _build_structured_raw_result as build_structured_result,
    )

    torch = torch_module
    assemble_molecular_graph = assemble_graph
    expand_molecule_graph = expand_graph
    recognize_rgroup_labels = recognize_labels
    _get_paddleocr_instances = get_paddleocr_instances
    _paddleocr_recognize = paddleocr_recognize
    _build_model = build_model
    _load_classes = load_classes
    _infer_runtime_from_config = infer_runtime
    _infer_single = infer_single
    _filter_sort_and_nms_boxes = filter_boxes
    _dedupe_lines_by_nearest_boxes = dedupe_lines
    _build_structured_raw_result = build_structured_result
    _inference_dependencies_loaded = True

def get_model():
    global _model, _postprocessor, _cfg, _box_classes, _line_classes
    global _image_size, _mean, _std, _classes_path, _device
    if _model is not None:
        return _model, _postprocessor, _image_size, _mean, _std, _box_classes, _line_classes, _classes_path

    _load_inference_dependencies()
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading I2BL TODO2 model on {_device}...")
    _cfg, _model, _postprocessor = _build_model(str(DEFAULT_CONFIG), str(DEFAULT_RESUME), _device)
    _image_size, _mean, _std, _classes_path, _ = _infer_runtime_from_config(_cfg)
    if os.environ.get("I2BL_CLASSES") or _classes_path is None or not Path(_classes_path).exists():
        _classes_path = DEFAULT_CLASSES
    _box_classes, _line_classes = _load_classes(str(_classes_path))
    print(f"Model loaded. Image size: {_image_size}, Classes: {_classes_path}")
    return _model, _postprocessor, _image_size, _mean, _std, _box_classes, _line_classes, _classes_path


def resolve_ketcher_dir():
    configured = os.environ.get("I2BL_KETCHER_DIR")
    if configured:
        return Path(configured)
    for candidate in KETCHER_CANDIDATES:
        if (candidate / "index.html").exists():
            return candidate
    return KETCHER_CANDIDATES[0]


def sanitize_stem(filename: str) -> str:
    stem = Path(filename or "image").stem
    safe = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in stem).strip("._")
    return safe or "image"


def build_unique_stem(filename: str) -> str:
    return f"{sanitize_stem(filename)}_{secrets.token_hex(3)}"


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + secrets.token_hex(4) + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


ATOM_LABEL_ALIASES = {
    "me": "CH3", "ch3": "CH3", "h3c": "CH3",
    "et": "C2H5", "c2h5": "C2H5", "h5c2": "C2H5",
}
BOND_LABEL_ALIASES = {
    "-": "single", "single bond": "single",
    "=": "double", "double bond": "double",
    "#": "triple", "≡": "triple", "triple bond": "triple",
    ":": "aromatic", "ar": "aromatic", "aromatic bond": "aromatic",
}


def _canonical_atom_label(label):
    label = str(label or "").strip()
    return ATOM_LABEL_ALIASES.get(label.casefold(), label)


def _canonical_bond_label(label):
    label = str(label or "").strip().casefold()
    return BOND_LABEL_ALIASES.get(label, label)


def _canonical_charge_label(label):
    value = str(label or "").strip().replace("−", "-")
    if value in {"+", "plus"}: return "+1"
    if value in {"-", "minus"}: return "-1"
    try:
        return f"{int(value):+d}"
    except ValueError:
        return value


def _detection_class_path():
    # Manual review/reassembly must work without loading any inference model.
    if DEFAULT_CLASSES.is_file():
        return DEFAULT_CLASSES
    if DEFAULT_INFERENCE_PROFILE == "dual_decoder":
        path = PROJECT_ROOT / "dataset_pair_real_20261015" / "annotations" / "classes.json"
        if path.is_file():
            return path
    _load_inference_dependencies()
    return Path(get_model()[-1])


def _graph_from_detection_payload(ac_payload, b_payload, image_size):
    """Reassemble an edited detector view, assigning each bond end to its nearest atom."""
    from engine.rtv4.chem_graph import assemble_molecular_graph, load_chem_class_maps
    from pair_training.hydrogen_notation import collapse_hydrogen_notation

    classes_path = _detection_class_path()
    atom_classes, bond_classes = load_chem_class_maps(classes_path)
    atom_ids = {name.casefold(): idx for idx, name in atom_classes.items()}
    charge_ids = {name: idx for idx, name in atom_classes.items() if name in {"-4", "-2", "-1", "+1", "+2"}}
    bond_ids = {}
    for idx, name in bond_classes.items():
        canonical = _canonical_bond_label(name)
        bond_ids[canonical] = idx

    boxes, labels, scores = [], [], []
    for item in ac_payload.get("items", []):
        label = str(item.get("label", "")).strip()
        source = item.get("source", "atom")
        if source == "charge":
            charge_label = _canonical_charge_label(label)
            class_id = charge_ids.get(charge_label)
            if class_id is None:
                continue
        else:
            canonical = _canonical_atom_label(label)
            class_id = atom_ids.get(canonical.casefold(), atom_ids.get("other", 0))
        box = [float(value) for value in item.get("bbox", [])]
        if len(box) != 4:
            continue
        boxes.append(box)
        labels.append(class_id)
        scores.append(float(item.get("score", 1.0)))

    lines, line_labels, line_scores = [], [], []
    for item in b_payload.get("items", []):
        line = [float(value) for value in item.get("bbox", [])]
        if len(line) != 4:
            continue
        canonical = _canonical_bond_label(item.get("label", "single"))
        class_id = bond_ids.get(canonical)
        if class_id is None:
            raise ValueError(f"Unsupported bond label: {item.get('label')}")
        lines.append(line)
        line_labels.append(class_id)
        line_scores.append(float(item.get("score", 1.0)))

    result = {
        "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": np.asarray(labels, dtype=np.int64),
        "scores": np.asarray(scores, dtype=np.float32),
        "lines": np.asarray(lines, dtype=np.float32).reshape(-1, 4),
        "line_labels": np.asarray(line_labels, dtype=np.int64),
        "line_scores": np.asarray(line_scores, dtype=np.float32),
    }
    graph = assemble_molecular_graph(result, classes_path=str(classes_path),
                                     box_score_thresh=0.0, line_score_thresh=0.0)

    # Retain the literal manual atom text for wildcard/R-group labels in Ketcher.
    atom_text_items = [item for item in ac_payload.get("items", []) if item.get("source") != "charge"]
    text_labels = {}
    for atom in graph.get("atoms", []):
        if atom.get("symbol") != "*":
            continue
        nearest = min(atom_text_items, key=lambda item: math.dist(
            atom.get("center", [0, 0]), item.get("center") or _bbox_center(item["bbox"])), default=None)
        if nearest:
            text_labels[atom["index"]] = {"text": str(nearest.get("label", "*"))}
    graph, text_labels, hydrogen_audit = collapse_hydrogen_notation(graph, text_labels)
    if hydrogen_audit:
        graph["hydrogen_notation"] = hydrogen_audit
    return graph, text_labels


def _normalized_atom_label(label):
    label = str(label or "").strip()
    return ATOM_LABEL_ALIASES.get(label.casefold(), label)


def _canonicalize_detection_ground_truth(ac_payload, b_payload, ocr_payload, *, image_size,
                                         image_file, image_sha256, source_stem, revision, saved_at_utc):
    from pair_training.hydrogen_notation import _notation_formula

    def canonical_items(items, kind):
        result = []
        for item in items:
            label = str(item.get("label", "")).strip()
            key = label.casefold()
            formula = _notation_formula(label) if kind == "atom" else None
            canonical = (formula[0] if formula else ATOM_LABEL_ALIASES.get(key, label)
                         if kind in {"atom", "charge"} else BOND_LABEL_ALIASES.get(key, label))
            if kind == "charge": canonical = _canonical_charge_label(label)
            normalized = dict(item)
            normalized["label"] = label
            normalized["canonical_label"] = canonical
            normalized["source"] = item.get("source", kind)
            if formula:
                normalized["hydrogens_in_notation"] = formula[1]
            result.append(normalized)
        return result

    ac_items = list(ac_payload.get("items", []))
    atoms = [item for item in ac_items if item.get("source") != "charge"]
    charges = [item for item in ac_items if item.get("source") == "charge"]
    return {
        "schema": "i2bl_detection_ground_truth_v1",
        "revision": revision,
        "saved_at_utc": saved_at_utc,
        "source_stem": source_stem,
        "image": {"file": image_file, "sha256": image_sha256, "size": image_size},
        "label_conventions": {
            "atom_aliases": {"Me": "CH3", "Et": "C2H5"},
            "hydrogen_notation": "heavy-atom + H labels are canonicalized to the heavy atom; valence is inferred from bonds and formal charge",
            "bond_aliases": {"=": "double", "-": "single", "≡": "triple"},
            "raw_label_is_preserved": True,
        },
        "atoms": canonical_items(atoms, "atom"),
        "charges": canonical_items(charges, "charge"),
        "bonds": canonical_items(list(b_payload.get("items", [])), "bond"),
        "ocr": ocr_payload,
    }


def _refresh_training_pairs_manifest():
    """Write a compact index for the human-corrected image/label collection."""
    samples = []
    if TRAINING_PAIRS_DIR.exists():
        for pair_dir in sorted(path for path in TRAINING_PAIRS_DIR.iterdir() if path.is_dir()):
            ground_truth_path = pair_dir / "ground_truth.json"
            image_path = pair_dir / "image.png"
            if not ground_truth_path.is_file() or not image_path.is_file():
                continue
            try:
                ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            revisions_dir = pair_dir / "revisions"
            revisions = sorted(p.name for p in revisions_dir.glob("revision_*.json")) if revisions_dir.exists() else []
            samples.append({
                "image_sha256": pair_dir.name,
                "image": str(image_path.relative_to(FIXED_DATA_DIR)),
                "ground_truth": str(ground_truth_path.relative_to(FIXED_DATA_DIR)),
                "source_stem": ground_truth.get("source_stem"),
                "revision": ground_truth.get("revision", 1),
                "saved_at_utc": ground_truth.get("saved_at_utc"),
                "image_size": ground_truth.get("image", {}).get("size", {}),
                "counts": {
                    "atoms": len(ground_truth.get("atoms", [])),
                    "charges": len(ground_truth.get("charges", [])),
                    "bonds": len(ground_truth.get("bonds", [])),
                    "ocr": len(ground_truth.get("ocr", {}).get("items", [])),
                },
                "revisions": revisions,
            })
    write_json(TRAINING_PAIRS_DIR / "manifest.json", {
        "schema": "i2bl_human_corrected_dataset_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "samples": samples,
    })


def _refresh_fixed_records_manifest():
    """Index the complete per-result Fix artifacts without scanning JSON manually."""
    records = []
    if FIXED_RECORDS_DIR.exists():
        for record_dir in sorted(path for path in FIXED_RECORDS_DIR.iterdir() if path.is_dir()):
            image_path = record_dir / "image.png"
            ac_path = record_dir / f"{record_dir.name}_ac.json"
            b_path = record_dir / f"{record_dir.name}_b.json"
            ocr_path = record_dir / f"{record_dir.name}_ocr.json"
            if not image_path.is_file():
                continue
            try:
                image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            except OSError:
                image_hash = None
            ac_items = b_items = ocr_items = 0
            try:
                ac_items = len(json.loads(ac_path.read_text(encoding="utf-8")).get("data", {}).get("items", []))
            except (OSError, json.JSONDecodeError):
                pass
            try:
                b_items = len(json.loads(b_path.read_text(encoding="utf-8")).get("data", {}).get("items", []))
            except (OSError, json.JSONDecodeError):
                pass
            try:
                ocr_items = len(json.loads(ocr_path.read_text(encoding="utf-8")).get("data", {}).get("items", []))
            except (OSError, json.JSONDecodeError):
                pass
            revision_dirs = sorted(p.name for p in (record_dir / "revisions").glob("revision_*") if p.is_dir())
            records.append({
                "result_stem": record_dir.name,
                "image_sha256": image_hash,
                "directory": str(record_dir.relative_to(FIXED_DATA_DIR)),
                "counts": {"atom_charge_items": ac_items, "bond_items": b_items, "ocr_items": ocr_items},
                "files": sorted(p.name for p in record_dir.iterdir() if p.is_file()),
                "revisions": revision_dirs,
            })
    write_json(FIXED_RECORDS_DIR / "manifest.json", {
        "schema": "i2bl_fix_records_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "records": records,
    })
    readme = FIXED_DATA_DIR / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Human corrected WebApp data\n\n"
            "This directory stores manually reviewed detections. Keep it as training data; "
            "do not treat it as a cache. Back up this directory.\n\n"
            "- `training_pairs/<sha256>/image.png`: deduplicated source image.\n"
            "- `training_pairs/<sha256>/ground_truth.json`: latest atom, charge, bond, and OCR labels.\n"
            "- `training_pairs/<sha256>/revisions/`: immutable per-Fix label history.\n"
            "- `training_pairs/manifest.json`: sample index and label counts.\n"
            "- `records/<result_stem>/`: per-result source image and raw Fix artifacts.\n"
            "- `records/<result_stem>/revisions/`: complete Fix snapshots with paired labels.\n"
            "- `records/manifest.json`: per-result index.\n\n"
            "The configured WebApp examples directory mirrors each image as `fixed_<hash>.png`; "
            "its same-stem JSON contains the latest labels. Use `training_pairs/` for training.\n",
            encoding="utf-8",
        )


def _safe_molblock(mol):
    """Serialize a molecule, repairing invalid predicted aromatic flags if needed."""
    if mol is None:
        return ""
    try:
        return Chem.MolToMolBlock(mol)
    except Exception:
        # A detector can occasionally label an isolated/non-ring atom or bond as
        # aromatic. RDKit refuses to write such a molecule. Preserve the graph as
        # an editable molfile by degrading aromatic bonds to single bonds.
        repaired = Chem.RWMol(mol)
        for bond in repaired.GetBonds():
            if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
                bond.SetIsAromatic(False)
                bond.SetBondType(Chem.BondType.SINGLE)
        for atom in repaired.GetAtoms():
            atom.SetIsAromatic(False)
        repaired_mol = repaired.GetMol()
        try:
            repaired_mol.UpdatePropertyCache(strict=False)
            return Chem.MolToMolBlock(repaired_mol, kekulize=False)
        except Exception as exc:
            print(f"[webapp] Failed to serialize repaired molfile: {exc}")
            return ""


def _extract_mol_graph(mol):
    graph = {"atoms": [], "bonds": []}
    if mol is None:
        return graph
    conformer = mol.GetConformer() if mol.GetNumConformers() else None
    for atom in mol.GetAtoms():
        coords = None
        if conformer is not None:
            pos = conformer.GetAtomPosition(atom.GetIdx())
            coords = {"x": pos.x, "y": pos.y, "z": pos.z}
        try:
            implicit_h_count = atom.GetNumImplicitHs()
        except Exception:
            implicit_h_count = None
        graph["atoms"].append({
            "idx": atom.GetIdx(),
            "symbol": atom.GetSymbol(),
            "formal_charge": atom.GetFormalCharge(),
            "explicit_h_count": atom.GetNumExplicitHs(),
            "implicit_h_count": implicit_h_count,
            "atom_label": atom.GetProp("atomLabel") if atom.HasProp("atomLabel") else None,
            "coords": coords,
        })
    for bond in mol.GetBonds():
        graph["bonds"].append({
            "idx": bond.GetIdx(),
            "begin": bond.GetBeginAtomIdx(),
            "end": bond.GetEndAtomIdx(),
            "bond_type": str(bond.GetBondType()),
            "bond_order": bond.GetBondTypeAsDouble(),
            "bond_dir": str(bond.GetBondDir()),
        })
    return graph


BOND_TYPE_MAP = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
    "aromatic": Chem.BondType.AROMATIC,
}
BOND_DIR_MAP = {
    "wedge": Chem.rdchem.BondDir.BEGINWEDGE,
    "dash": Chem.rdchem.BondDir.BEGINDASH,
}


def _reverse_bond_dir(direction):
    if direction == Chem.rdchem.BondDir.BEGINWEDGE:
        return Chem.rdchem.BondDir.BEGINDASH
    if direction == Chem.rdchem.BondDir.BEGINDASH:
        return Chem.rdchem.BondDir.BEGINWEDGE
    return direction


def _add_2d_conformer(mol, graph, rd_map):
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom in graph.get("atoms", []):
        rd_idx = rd_map.get(atom["index"])
        center = atom.get("center") or [0.0, 0.0]
        if rd_idx is not None:
            conformer.SetAtomPosition(rd_idx, (float(center[0]) / 100.0, float(-center[1]) / 100.0, 0.0))
    mol.RemoveAllConformers()
    mol.AddConformer(conformer)


def _apply_wedge_dash_bond_dirs(mol, directional_bonds):
    if not directional_bonds:
        return
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    chiral_centers = {idx for idx, _ in Chem.FindMolChiralCenters(mol, includeUnassigned=True)}
    for mapped_begin, mapped_end, stereo in directional_bonds:
        bond = mol.GetBondBetweenAtoms(mapped_begin, mapped_end)
        if bond is None:
            continue
        desired_dir = BOND_DIR_MAP.get(stereo)
        if desired_dir is None:
            continue
        current_begin = bond.GetBeginAtomIdx()
        reverse_dir = _reverse_bond_dir(desired_dir)
        if mapped_begin in chiral_centers or mapped_end in chiral_centers:
            chiral_atom = mapped_begin if mapped_begin in chiral_centers else mapped_end
            bond.SetBondDir(desired_dir if current_begin == chiral_atom else reverse_dir)
        else:
            bond.SetBondDir(desired_dir if current_begin == mapped_begin else reverse_dir)
    try:
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        pass


def _mol_from_i2bl_graph(graph, ocr_labels=None):
    ocr_labels = ocr_labels or {}
    rwmol = Chem.RWMol()
    rd_map = {}
    directional_bonds = []
    for atom in graph.get("atoms", []):
        symbol = atom.get("symbol", "*")
        if symbol == "*":
            rd_atom = Chem.Atom(0)
            label = str(ocr_labels.get(atom["index"], {}).get("text") or "*")
            rd_atom.SetProp("atomLabel", label)
            rd_atom.SetProp("pseudoAtom", label)
            rd_atom.SetProp("molFileAlias", label)
        else:
            try:
                rd_atom = Chem.Atom(symbol)
            except Exception:
                rd_atom = Chem.Atom(0)
                rd_atom.SetProp("atomLabel", symbol)
            rd_atom.SetFormalCharge(int(atom.get("formal_charge", 0)))
        rd_map[atom["index"]] = rwmol.AddAtom(rd_atom)

    for bond in graph.get("bonds", []):
        begin = rd_map.get(bond.get("begin_atom_idx"))
        end = rd_map.get(bond.get("end_atom_idx"))
        if begin is None or end is None or begin == end:
            continue
        graph_bond = bond.get("graph_bond") or {}
        kind = graph_bond.get("kind", "single")
        bond_type = BOND_TYPE_MAP.get(kind, Chem.BondType.SINGLE)
        if rwmol.GetBondBetweenAtoms(begin, end) is None:
            rwmol.AddBond(begin, end, bond_type)
            stereo = graph_bond.get("stereo")
            if stereo in BOND_DIR_MAP:
                directional_bonds.append((begin, end, stereo))

    mol = rwmol.GetMol()
    _add_2d_conformer(mol, graph, rd_map)
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        pass
    _apply_wedge_dash_bond_dirs(mol, directional_bonds)
    return mol


def _mol_from_smiles_for_ketcher(smiles):
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            AllChem.Compute2DCoords(mol)
        return mol
    except Exception:
        return None


def _image_size_payload(image):
    width, height = image.size
    return {"width": int(width), "height": int(height)}


def _items_to_group(items):
    return {
        "bbox": [item["bbox"] for item in items],
        "bbox_centers": [item.get("center") or _bbox_center(item["bbox"]) for item in items],
        "scores": [item.get("score", 0.0) for item in items],
        "pred_classes": [item.get("label", "other") for item in items],
    }


def _ocr_payload_from_result(ocr_result):
    items = []
    for idx, item in enumerate(ocr_result.get("ocr_atoms", []) or []):
        box = item.get("box") or [0, 0, 0, 0]
        text = item.get("text") or ""
        score = float(item.get("score") or item.get("confidence") or 0.0)
        items.append({
            "uid": f"ocr:{idx}",
            "det_index": int(item.get("atom_index", idx)),
            "bbox": box,
            "center": _bbox_center(box),
            "matched_index": int(item.get("atom_index", idx)),
            "original_text": text,
            "original_score": score,
            "scaled_text": text,
            "scaled_score": score,
            "label": text,
            "score": score,
        })
    return {"items": items}


def _bbox_iou(a, b):
    if not a or not b: return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


def _bbox_center(box):
    return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]


def _merge_fragmented_atom_boxes(items, image_size):
    """Merge adjacent predictions that cover separate strokes of one atom label.

    Implicit carbon detections represent molecular vertices and must remain
    separate. Explicit labels can produce adjacent, low-IoU fragments which
    ordinary NMS cannot remove.
    """
    if not items:
        return []

    parent = list(range(len(items)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    def should_merge(left, right):
        if left.get("label_name") != right.get("label_name"):
            return False
        # Carbon boxes mark distinct molecular vertices. A distance-based
        # union can join an entire ring/chain through intermediate vertices.
        # Duplicate detections of a vertex are already handled by NMS.
        if left.get("label_name") == "C":
            return False
        a, b = left["bbox_xyxy"], right["bbox_xyxy"]
        aw, ah = max(1.0, a[2] - a[0]), max(1.0, a[3] - a[1])
        bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
        x_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        y_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        x_gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
        y_gap = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
        gap_limit = max(2.0, 0.35 * max(min(aw, bw), min(ah, bh)))
        vertical_fragments = x_overlap >= 0.45 * min(aw, bw) and y_gap <= gap_limit
        horizontal_fragments = y_overlap >= 0.45 * min(ah, bh) and x_gap <= gap_limit
        return vertical_fragments or horizontal_fragments

    for left_index, left in enumerate(items):
        for right_index in range(left_index + 1, len(items)):
            if should_merge(left, items[right_index]):
                union(left_index, right_index)

    groups = {}
    for index, item in enumerate(items):
        groups.setdefault(find(index), []).append(item)

    image_width, image_height = image_size
    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(dict(group[0]))
            continue
        best = max(group, key=lambda item: float(item.get("score") or 0.0))
        box = [
            min(item["bbox_xyxy"][0] for item in group),
            min(item["bbox_xyxy"][1] for item in group),
            max(item["bbox_xyxy"][2] for item in group),
            max(item["bbox_xyxy"][3] for item in group),
        ]
        padding = max(1.0, 0.06 * max(box[2] - box[0], box[3] - box[1]))
        box = [
            max(0.0, box[0] - padding),
            max(0.0, box[1] - padding),
            min(float(image_width), box[2] + padding),
            min(float(image_height), box[3] + padding),
        ]
        combined = dict(best)
        combined["bbox_xyxy"] = box
        combined["center_xy"] = _bbox_center(box)
        combined["merged_fragment_count"] = len(group)
        combined["merged_raw_indices"] = [item.get("raw_index") for item in group]
        merged.append(combined)

    # Resolve competing explicit labels after their fragments have been joined.
    # Their union boxes often overlap strongly even though the original stroke
    # boxes did not. An explicit label also replaces an implicit carbon at the
    # same location.
    explicit = sorted(
        (item for item in merged if item.get("label_name") != "C"),
        key=lambda item: -float(item.get("score") or 0.0),
    )
    kept_explicit = []
    for item in explicit:
        if any(_bbox_iou(item["bbox_xyxy"], kept["bbox_xyxy"]) >= 0.45 for kept in kept_explicit):
            continue
        kept_explicit.append(item)

    # A single surviving fragment should still cover the whole printed label.
    # Keep the predicted center, but enforce a conservative label-shaped extent.
    for item in kept_explicit:
        label = str(item.get("label_name") or "")
        if label in {"other", "I"} or label.startswith(("+", "-")):
            continue
        box = item["bbox_xyxy"]
        center_x, center_y = _bbox_center(box)
        width = max(1.0, box[2] - box[0])
        height = max(1.0, box[3] - box[1])
        if len(label) == 1:
            height = max(height, width * 1.25)
        elif len(label) == 2:
            width = max(width, height * 1.05)
        padding = max(1.0, 0.05 * max(width, height))
        half_width = width / 2.0 + padding
        half_height = height / 2.0 + padding
        item["bbox_xyxy"] = [
            max(0.0, center_x - half_width),
            max(0.0, center_y - half_height),
            min(float(image_width), center_x + half_width),
            min(float(image_height), center_y + half_height),
        ]
        item["center_xy"] = _bbox_center(item["bbox_xyxy"])

    implicit_carbons = []
    for item in merged:
        if item.get("label_name") != "C":
            continue
        center = item["center_xy"]
        covered_by_label = any(
            label["bbox_xyxy"][0] <= center[0] <= label["bbox_xyxy"][2]
            and label["bbox_xyxy"][1] <= center[1] <= label["bbox_xyxy"][3]
            for label in kept_explicit
        )
        if not covered_by_label:
            implicit_carbons.append(item)

    merged = implicit_carbons + kept_explicit
    merged.sort(key=lambda item: (item["center_xy"][0], item["center_xy"][1], item.get("raw_index", -1)))
    for atom_index, item in enumerate(merged):
        item["atom_index"] = atom_index
    return merged


def _detect_full_image_text_regions(image):
    """Detect full printed labels with the configured available OCR backend."""
    try:
        if os.environ.get("I2BL_OCR_BACKEND") == "easyocr":
            from engine.rtv4.chem_ocr_expand import _get_easyocr_reader
            reader = _get_easyocr_reader(gpu=False)
            if reader is None: return []
            result = reader.readtext(np.asarray(image.convert("RGB")), detail=1)
            return [{"bbox": [min(p[0] for p in quad), min(p[1] for p in quad),
                              max(p[0] for p in quad), max(p[1] for p in quad)],
                     "text": str(text), "score": float(score)} for quad, text, score in result]
        if not hasattr(np, "int"):
            np.int = int
        primary, _ = _get_paddleocr_instances(use_gpu=False)
        if primary is None:
            return []
        result = primary.ocr(np.asarray(image.convert("RGB")), cls=True) or []
    except Exception as exc:
        print(f"[webapp] Full-image OCR detection failed: {exc}")
        return []

    if len(result) == 1 and isinstance(result[0], list) and result[0] and len(result[0]) != 2:
        result = result[0]
    regions = []
    for entry in result:
        try:
            quad, recognition = entry
            text, score = recognition
            xs = [float(point[0]) for point in quad]
            ys = [float(point[1]) for point in quad]
            regions.append({
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "text": str(text),
                "score": float(score),
            })
        except (TypeError, ValueError, IndexError):
            continue
    return regions


def _ocr_text_to_atom_label(text, available_labels):
    cleaned = "".join(ch for ch in str(text).strip() if not ch.isspace())
    tokens = re.findall(r"[A-Z][a-z]?", cleaned)
    valid = []
    periodic_table = Chem.GetPeriodicTable()
    for token in tokens:
        try:
            if periodic_table.GetAtomicNumber(token) > 0:
                valid.append(token)
        except Exception:
            continue
    non_hydrogen = [token for token in valid if token != "H"]
    if len(set(non_hydrogen)) == 1:
        label = non_hydrogen[0]
    elif len(valid) == 1:
        label = valid[0]
    else:
        label = "other"
    return label if label in available_labels else "other"


def _recognize_dense_text_cluster(image, box):
    width, height = image.size
    margin = max(5.0, 0.12 * max(box[2] - box[0], box[3] - box[1]))
    crop_box = [
        max(0, int(np.floor(box[0] - margin))),
        max(0, int(np.floor(box[1] - margin))),
        min(width, int(np.ceil(box[2] + margin))),
        min(height, int(np.ceil(box[3] + margin))),
    ]
    try:
        primary, secondary = _get_paddleocr_instances(use_gpu=False)
        crop = np.asarray(image.crop(crop_box).convert("RGB"))
        candidates = [
            _paddleocr_recognize(ocr, crop)
            for ocr in (primary, secondary)
            if ocr is not None
        ]
        text, score = max(candidates, key=lambda item: float(item[1] or 0.0))
        display_box = [
            max(0.0, float(box[0]) - 2.0),
            max(0.0, float(box[1]) - 2.0),
            min(float(width), float(box[2]) + 2.0),
            min(float(height), float(box[3]) + 2.0),
        ]
        return {"bbox": display_box, "text": text, "score": float(score or 0.0)}
    except Exception as exc:
        print(f"[webapp] Dense-cluster OCR failed: {exc}")
        return None


def _text_region_spans_predicted_bond(items, bond_lines):
    """A text region covering both ends of a bond contains separate atoms."""
    if len(items) < 2:
        return False
    for line in bond_lines:
        endpoints = (line[:2], line[2:])
        nearest = [min(range(len(items)), key=lambda index: float(np.hypot(
            point[0] - items[index]["center_xy"][0],
            point[1] - items[index]["center_xy"][1],
        ))) for point in endpoints]
        if nearest[0] == nearest[1]:
            continue
        if all(
            items[index]["bbox_xyxy"][0] - 2.0 <= point[0] <= items[index]["bbox_xyxy"][2] + 2.0
            and items[index]["bbox_xyxy"][1] - 2.0 <= point[1] <= items[index]["bbox_xyxy"][3] + 2.0
            for index, point in zip(nearest, endpoints)
        ):
            return True
    return False


def _replace_boxes_with_text_regions(items, regions, box_classes, image_size, bond_lines=None):
    """Replace stroke-sized predictions inside OCR regions with one label box."""
    if not items or not regions:
        return items
    inverse_classes = {name: index for index, name in box_classes.items()}
    remaining = list(items)
    replacements = []
    for region in sorted(regions, key=lambda item: -float(item.get("score") or 0.0)):
        if float(region.get("score") or 0.0) < 0.5:
            continue
        box = [float(value) for value in region["bbox"]]
        margin = max(3.0, 0.12 * max(box[2] - box[0], box[3] - box[1]))
        expanded = [box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin]
        matched = [
            item for item in remaining
            if expanded[0] <= item["center_xy"][0] <= expanded[2]
            and expanded[1] <= item["center_xy"][1] <= expanded[3]
        ]
        if not matched:
            continue
        # OCR can read an entire A—O or N—N fragment as one word. Keep the
        # detector's separate vertices when a confident bond connects them.
        if _text_region_spans_predicted_bond(matched, bond_lines or []):
            continue
        remaining = [item for item in remaining if item not in matched]
        label = _ocr_text_to_atom_label(region.get("text", ""), set(inverse_classes))
        best = max(matched, key=lambda item: float(item.get("score") or 0.0))
        replacement = dict(best)
        replacement["label_name"] = label
        replacement["label_id"] = inverse_classes.get(label, inverse_classes.get("other", 0))
        replacement["bbox_xyxy"] = [
            max(0.0, box[0] - 1.0),
            max(0.0, box[1] - 1.0),
            min(float(image_size[0]), box[2] + 1.0),
            min(float(image_size[1]), box[3] + 1.0),
        ]
        replacement["center_xy"] = _bbox_center(replacement["bbox_xyxy"])
        replacement["score"] = max(float(best.get("score") or 0.0), float(region.get("score") or 0.0))
        replacement["ocr_text"] = region.get("text", "")
        replacement["ocr_score"] = float(region.get("score") or 0.0)
        replacement["merged_fragment_count"] = sum(int(item.get("merged_fragment_count", 1)) for item in matched)
        replacements.append(replacement)

    output = remaining + replacements
    output.sort(key=lambda item: (item["center_xy"][0], item["center_xy"][1], item.get("raw_index", -1)))
    for atom_index, item in enumerate(output):
        item["atom_index"] = atom_index
    return output


def _bbox_center_distance(a, b):
    ca, cb = _bbox_center(a), _bbox_center(b)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def _is_close_bbox(a, b, tol=0.6):
    if a is None or b is None: return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def _match_atom_item_by_bbox(atom_items, target_bbox):
    best, best_score = None, None
    for item in atom_items:
        bbox = item.get("bbox")
        if not bbox: continue
        if _is_close_bbox(bbox, target_bbox): return item
        score = (_bbox_iou(bbox, target_bbox), -_bbox_center_distance(bbox, target_bbox))
        if best_score is None or score > best_score:
            best_score, best = score, item
    return best


def _is_valid_atom_label(label: str) -> bool:
    atom_sym, _ = _parse_atom_label(str(label))
    if atom_sym in {"*", "other"}: return True
    try: return Chem.GetPeriodicTable().GetAtomicNumber(atom_sym) > 0
    except: return False


def _parse_atom_label(label: str):
    """Parse the compact atom/charge labels used by the detector."""
    if "other" in label:
        return "*", -1 if label.endswith("-") else 1 if label.endswith("+") else 0
    if label.endswith("+"):
        return label[:-1], 1
    if label.endswith("-"):
        return label[:-1], -1
    if label and label[-1].isdigit():
        return label[:-1], int(label[-1])
    return label, 0


def _label_to_prediction_class_index(label: str) -> int:
    _, _, _, _box_classes, _, _, _, _ = get_model()
    inverse = {name: idx for idx, name in _box_classes.items()}
    if label in inverse: return inverse[label]
    if _is_valid_atom_label(label): return inverse.get("other", 0)
    raise ValueError(f"Unsupported label: {label}")


def _build_output_from_fix_payload(ac_payload, b_payload):
    items = list(ac_payload.get("items", [])) + list(b_payload.get("items", []))
    if not items: raise ValueError("No edited detection items")
    labels, boxes, scores = [], [], []
    for item in items:
        labels.append(_label_to_prediction_class_index(item["label"]))
        boxes.append(item["bbox"])
        scores.append(item.get("score", 1))
    boxes_np = np.array(boxes, dtype=np.float32)
    centers = np.column_stack(((boxes_np[:, 0] + boxes_np[:, 2]) / 2, (boxes_np[:, 1] + boxes_np[:, 3]) / 2))
    return {"bbox": boxes_np, "bbox_centers": centers, "scores": np.array(scores, dtype=np.float32),
            "pred_classes": np.array(labels, dtype=np.int64)}


def _filter_charge_boxes(raw_result, box_classes, score_cutoff, nms_iou):
    """Keep charge detections separately from atom NMS."""
    charge_labels = {"-4", "-2", "-1", "+1", "+2"}
    charges = []
    for raw_index, (box, label, score) in enumerate(zip(
        raw_result["boxes"], raw_result["labels"], raw_result["scores"]
    )):
        label_id = int(label)
        score = float(score)
        if box_classes.get(label_id) not in charge_labels or score < score_cutoff:
            continue
        box = [float(value) for value in box]
        charges.append({
            "raw_index": raw_index,
            "label_id": label_id,
            "label_name": box_classes[label_id],
            "score": score,
            "bbox_xyxy": box,
            "center_xy": _bbox_center(box),
        })
    # Charge symbols on the same atom may carry different classes. Keep the
    # strongest overlapping prediction; atom NMS must not suppress charges.
    kept = []
    for item in sorted(charges, key=lambda entry: -entry["score"]):
        if all(_bbox_iou(item["bbox_xyxy"], prior["bbox_xyxy"]) <= nms_iou for prior in kept):
            kept.append(item)
    return kept


def _run_i2bl_inference(
    image_path: str,
    atom_cutoff=DEFAULT_ATOM_CUTOFF,
    bond_cutoff=DEFAULT_LINE_CUTOFF,
    atom_nms=DEFAULT_ATOM_NMS,
    charge_cutoff=DEFAULT_ATOM_CUTOFF,
    charge_nms=DEFAULT_CHARGE_NMS,
):
    """Run I2BL TODO2 inference on an image."""
    if DEFAULT_INFERENCE_PROFILE == "dual_decoder":
        if not DEFAULT_PAIR_BUNDLE.is_dir():
            raise FileNotFoundError(f"Dual-decoder bundle not found: {DEFAULT_PAIR_BUNDLE}")
        return _run_dual_decoder_inference(image_path)
    if DEFAULT_INFERENCE_PROFILE != "legacy":
        raise ValueError(f"Unsupported I2BL_INFERENCE_PROFILE: {DEFAULT_INFERENCE_PROFILE}")
    _load_inference_dependencies()
    model, postprocessor, image_size, mean, std, box_classes, line_classes, classes_path = get_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image, raw_result = _infer_single(model, postprocessor, image_path, device, image_size, mean, std)

    # Filter boxes
    filtered_boxes = _filter_sort_and_nms_boxes(raw_result, box_classes, score_thr=atom_cutoff, nms_iou_thr=atom_nms)
    filtered_boxes = _merge_fragmented_atom_boxes(filtered_boxes, image.size)
    text_regions = _detect_full_image_text_regions(image)
    for item in filtered_boxes:
        if item.get("label_name") == "C" and int(item.get("merged_fragment_count", 1)) >= 3:
            region = _recognize_dense_text_cluster(image, item["bbox_xyxy"])
            if region is not None:
                text_regions.append(region)
    confident_bond_lines = [
        line.tolist() if hasattr(line, "tolist") else line
        for line, score in zip(raw_result["lines"], raw_result["line_scores"])
        if float(score) >= DEFAULT_LINE_CUTOFF
    ]
    filtered_boxes = _replace_boxes_with_text_regions(
        filtered_boxes, text_regions, box_classes, image.size, bond_lines=confident_bond_lines,
    )
    charge_boxes = _filter_charge_boxes(raw_result, box_classes, charge_cutoff, charge_nms)

    # Build structured raw result for atom/charge/bond display
    kept_lines = _dedupe_lines_by_nearest_boxes(raw_result, filtered_boxes, line_classes, line_score_thr=bond_cutoff)
    # Assemble the molecule from the same detections shown in the editors.
    # The previous raw_result path silently rebuilt the discarded fragments.
    processed_result = _build_structured_raw_result(filtered_boxes + charge_boxes, kept_lines)
    graph = assemble_molecular_graph(
        processed_result, classes_path=str(classes_path),
        box_score_thresh=0.0, line_score_thresh=0.0,
    )

    # OCR expansion
    ocr_labels = recognize_rgroup_labels(graph, image, use_gpu=torch.cuda.is_available(),
                                       backend=os.environ.get("I2BL_OCR_BACKEND", "auto"))
    ocr_result = expand_molecule_graph(graph, ocr_labels)
    ocr_result["ocr_labels"] = ocr_labels

    # Generate SMILES
    skeleton_smiles = ""
    expanded_smiles = ""
    if ocr_result.get("skeleton_smiles"):
        skeleton_smiles = ocr_result["skeleton_smiles"]
    if ocr_result.get("expanded_smiles"):
        expanded_smiles = ocr_result["expanded_smiles"]

    # Generate Ketcher-ready Molfiles. The graph-derived molecule keeps
    # wedge/dash bond directions, unlike maybe_build_rdkit().
    skeleton_mol = _mol_from_i2bl_graph(graph, ocr_labels)
    expanded_mol = _mol_from_smiles_for_ketcher(expanded_smiles) or skeleton_mol
    preferred_mol = expanded_mol or skeleton_mol
    molfile = {
        "skeleton": _safe_molblock(skeleton_mol),
        "expanded": _safe_molblock(expanded_mol),
        "preferred": _safe_molblock(preferred_mol),
    }
    graph_payload = {
        "skeleton": _extract_mol_graph(skeleton_mol),
        "expanded": _extract_mol_graph(expanded_mol),
        "preferred": _extract_mol_graph(preferred_mol),
    }

    # Build detection payloads (for frontend display)
    atom_items = []
    for box_info in filtered_boxes + charge_boxes:
        label_name = box_classes.get(box_info['label_id'], 'other')
        atom_items.append({
            "bbox": box_info['bbox_xyxy'],
            "center": _bbox_center(box_info['bbox_xyxy']),
            "label": label_name,
            "score": float(box_info['score']) if box_info['score'] is not None else 0.0,
            "source": "charge" if label_name in {"-4", "-2", "-1", "+1", "+2"} else "atom",
        })

    bond_items = []
    for line_info in kept_lines:
        label_name = line_classes.get(line_info['label_id'], 'other')
        bond_items.append({
            "bbox": line_info['line_xyxy'],
            "center": _bbox_center(line_info['line_xyxy']),
            "label": label_name,
            "score": float(line_info['score']) if line_info['score'] is not None else 0.0,
            "source": "bond",
        })
    atom_only_items = [item for item in atom_items if item["source"] == "atom"]
    charge_items = [item for item in atom_items if item["source"] == "charge"]

    return {
        "image": image,
        "raw_result": raw_result,
        "graph": graph,
        "ketcher_graph": graph_payload,
        "ocr_result": ocr_result,
        "ocr": _ocr_payload_from_result(ocr_result),
        "skeleton_smiles": skeleton_smiles,
        "expanded_smiles": expanded_smiles,
        "molfile": molfile,
        "atom_items": atom_items,
        "bond_items": bond_items,
        "detections": {
            "atoms": _items_to_group(atom_only_items),
            "charges": _items_to_group(charge_items),
            "bonds": _items_to_group(bond_items),
        },
        "image_path": image_path,
    }


def _run_dual_decoder_inference(image_path: str):
    """Use the best validated bundle: learned text entities + anchors + pair bonds + adapted OCR."""
    from pair_bundle_backend import predict_pair_bundle

    prediction = predict_pair_bundle(image_path, DEFAULT_PAIR_BUNDLE, PAIR_DEVICE)
    image = prediction["image"]
    graph = prediction["graph"]
    ocr_labels = prediction["labels"]
    skeleton_smiles = prediction.get("cxsmiles") or ""
    expanded_smiles = prediction.get("expanded_smiles") or ""

    skeleton_mol = _mol_from_i2bl_graph(graph, ocr_labels)
    expanded_mol = _mol_from_smiles_for_ketcher(expanded_smiles) or skeleton_mol
    preferred_mol = expanded_mol or skeleton_mol
    molfile = {
        "skeleton": _safe_molblock(skeleton_mol),
        "expanded": _safe_molblock(expanded_mol),
        "preferred": _safe_molblock(preferred_mol),
    }
    graph_payload = {
        "skeleton": _extract_mol_graph(skeleton_mol),
        "expanded": _extract_mol_graph(expanded_mol),
        "preferred": _extract_mol_graph(preferred_mol),
    }

    atom_items, charge_items = [], []
    for atom in graph.get("atoms", []):
        box = atom.get("box") or [0, 0, 0, 0]
        label = atom.get("atom_notation") or (
            ocr_labels.get(atom["index"], {}).get("text", "*") if atom.get("symbol") == "*"
            else atom.get("symbol", "C"))
        atom_items.append({"bbox": box, "center": atom.get("center") or _bbox_center(box),
                           "label": label, "score": float(atom.get("score", 0.0)), "source": "atom"})
        for source in atom.get("charge_sources", []):
            charge_box = source.get("box") or box
            charge_items.append({"bbox": charge_box, "center": _bbox_center(charge_box),
                                 "label": source.get("label_name", "charge"),
                                 "score": float(source.get("score", 0.0)), "source": "charge"})

    bond_items = []
    for bond in graph.get("bonds", []):
        line = bond.get("line") or [0, 0, 0, 0]
        # Bonds are oriented line segments, not rectangles. Preserve both
        # detector endpoints; min/max normalization flips negative-slope lines.
        endpoints = [float(value) for value in line]
        center = [(endpoints[0] + endpoints[2]) / 2, (endpoints[1] + endpoints[3]) / 2]
        bond_items.append({"bbox": endpoints, "center": center,
                           "label": bond.get("label_name", "bond"),
                           "score": float(bond.get("score", 0.0)), "source": "bond"})

    ocr_atoms = []
    placeholder_atoms = {}
    for atom_index, label in sorted(ocr_labels.items()):
        item = {"atom_index": int(atom_index), "text": label.get("raw_text") or label.get("text", ""),
                "score": float(label.get("score", 0.0)), "box": label.get("box") or [],
                "smiles": label.get("smiles", "[*]"), "scale": label.get("scale"),
                "backend": prediction.get("text_recognizer")}
        ocr_atoms.append(item)
        placeholder_atoms[str(atom_index)] = item["text"]
    ocr_result = {
        "skeleton_smiles": skeleton_smiles,
        "expanded_smiles": expanded_smiles,
        "ocr_atoms": ocr_atoms,
        "placeholder_atoms": placeholder_atoms,
        "backend": prediction.get("text_recognizer"),
        "entity_route": graph.get("text_entity_route", {}).get("selected", "learner"),
        "region_audit": prediction.get("region_audit"),
        "anchor_audit": prediction.get("anchor_audit"),
        "hydrogen_notation": prediction.get("hydrogen_notation", []),
        "chemistry": prediction.get("chemistry"),
    }
    atom_only_items = [item for item in atom_items if item["source"] == "atom"]
    return {
        "image": image,
        "raw_result": {},
        "graph": graph,
        "ketcher_graph": graph_payload,
        "ocr_result": ocr_result,
        "ocr": _ocr_payload_from_result(ocr_result),
        "skeleton_smiles": skeleton_smiles,
        "expanded_smiles": expanded_smiles,
        "molfile": molfile,
        "atom_items": atom_only_items + charge_items,
        "bond_items": bond_items,
        "detections": {
            "atoms": _items_to_group(atom_only_items),
            "charges": _items_to_group(charge_items),
            "bonds": _items_to_group(bond_items),
        },
        "image_path": image_path,
        "inference_profile": "dual_decoder_pilot_20261022",
    }


def save_prediction_bundle(result, image_filename: str, source="upload", source_image_path=None):
    """Save all prediction results to disk."""
    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_stem = build_unique_stem(image_filename)
    image_path = RESULTS_DATA_DIR / f"{save_stem}.png"

    # Save image
    if source_image_path is not None:
        shutil.copy2(source_image_path, image_path)
    else:
        result["image"].save(image_path)
    image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    pasted_example = None
    if source == "clipboard":
        PASTED_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        pasted_example = f"paste_{image_digest[:16]}.png"
        pasted_example_path = PASTED_EXAMPLES_DIR / pasted_example
        if not pasted_example_path.exists():
            shutil.copy2(image_path, pasted_example_path)

    # Detection payloads
    ac_payload = {"items": result["atom_items"]}
    b_payload = {"items": result["bond_items"]}

    image_size = _image_size_payload(result["image"])
    smiles = {"skeleton": result["skeleton_smiles"], "expanded": result["expanded_smiles"]}
    placeholder_atoms = result["ocr_result"].get("placeholder_atoms", {})

    # SMILES payload
    smiles_payload = {
        "stem": save_stem,
        "image_file": image_path.name,
        "image_sha256": image_digest,
        "annotation_provenance": result.get("annotation_provenance"),
        "smiles": smiles,
        "placeholder_atoms": placeholder_atoms,
    }

    molfile_files = {
        "skeleton": f"{save_stem}_original.mol",
        "expanded": f"{save_stem}_expanded.mol",
        "preferred": f"{save_stem}_preferred.mol",
    }
    graph_files = {
        "skeleton": f"{save_stem}_original_graph.json",
        "expanded": f"{save_stem}_expanded_graph.json",
        "preferred": f"{save_stem}_preferred_graph.json",
    }
    for key, filename in molfile_files.items():
        (RESULTS_DATA_DIR / filename).write_text(result["molfile"].get(key) or "", encoding="utf-8")
    for key, filename in graph_files.items():
        write_json(RESULTS_DATA_DIR / filename, result["ketcher_graph"].get(key) or {})

    # Prediction record
    prediction_record = {
        "image_file": image_path.name,
        "result_stem": save_stem,
        "upload_source": source,
        "pasted_example_file": pasted_example,
        "inference_profile": result.get("inference_profile", DEFAULT_INFERENCE_PROFILE),
        "smiles": smiles,
        "placeholder_atoms": placeholder_atoms,
        "image_size": image_size,
        "image_sha256": image_digest,
        "annotation_provenance": result.get("annotation_provenance"),
        "ocr": result["ocr"],
        "ocr_raw": result["ocr_result"],
        "detection_files": {"ac": f"{save_stem}_ac.json", "b": f"{save_stem}_b.json"},
        "molfile_files": molfile_files,
        "graph_files": graph_files,
    }

    write_json(RESULTS_DATA_DIR / f"{save_stem}_smiles.json", smiles_payload)
    write_json(RESULTS_DATA_DIR / f"{save_stem}_prediction.json", prediction_record)
    write_json(RESULTS_DATA_DIR / f"{save_stem}_ac.json", ac_payload)
    write_json(RESULTS_DATA_DIR / f"{save_stem}_b.json", b_payload)
    write_json(RESULTS_DATA_DIR / f"{save_stem}_ocr.json", result["ocr"])

    result["result_stem"] = save_stem
    return {
        "result_stem": save_stem,
        "inference_profile": result.get("inference_profile", DEFAULT_INFERENCE_PROFILE),
        "image_size": image_size,
        "image_sha256": image_digest,
        "smiles": smiles,
        "molfile": result["molfile"],
        "graph": result["ketcher_graph"],
        "detections": result["detections"],
        "ocr": result["ocr"],
        "storage": {
            "results_data_dir": str(RESULTS_DATA_DIR),
            "files": {
                "image": image_path.name,
                **({"pasted_example": pasted_example} if pasted_example else {}),
                "smiles": f"{save_stem}_smiles.json",
                "prediction": f"{save_stem}_prediction.json",
                "ac_json": f"{save_stem}_ac.json",
                "b_json": f"{save_stem}_b.json",
                "ocr_json": f"{save_stem}_ocr.json",
                **{f"{key}_molfile": value for key, value in molfile_files.items()},
                **{f"{key}_graph": value for key, value in graph_files.items()},
            },
        },
    }


def _reassemble_detection_payload(ac_payload, b_payload):
    graph, ocr_labels = _graph_from_detection_payload(ac_payload, b_payload, ac_payload.get("image_size", {}))
    for atom in graph.get("atoms", []):
        notation = atom.get("atom_notation")
        if notation and atom["index"] not in ocr_labels:
            ocr_labels[atom["index"]] = {"text": notation, "raw_text": notation}
    mol = _mol_from_i2bl_graph(graph, ocr_labels)
    molfile = _safe_molblock(mol)
    graph_mol = _extract_mol_graph(mol)
    from engine.rtv4.chem_ocr_expand import ABBREVIATIONS, ELEMENTS, RGROUP_SYMBOLS, expand_abbreviation
    from pair_training.notebook_ocr import expand_labels, labeled_cxsmiles
    manual_labels = {}
    for index, value in ocr_labels.items():
        text = str(value.get("raw_text") or value.get("text") or "").strip()
        supported = text in ABBREVIATIONS or text in ELEMENTS
        manual_labels[int(index)] = {
            "text": text,
            "raw_text": text,
            "score": 1.0,
            "eligible_for_expansion": supported and text not in RGROUP_SYMBOLS,
            "smiles": expand_abbreviation(text),
            "merged_atom_indices": [int(index)],
        }
    _, cx_result = labeled_cxsmiles(mol, graph, manual_labels)
    expanded_mol, expansion_result = expand_labels(mol, graph, manual_labels)
    try:
        fallback_smiles = Chem.MolToSmiles(mol)
    except Exception:
        fallback_smiles = ""
    skeleton_smiles = cx_result.get("cxsmiles") or fallback_smiles
    expanded_smiles = (expansion_result.get("smiles") if expansion_result.get("valid") else None) or fallback_smiles
    return {
        "smiles": {"skeleton": skeleton_smiles, "expanded": expanded_smiles},
        "molfile": {"skeleton": molfile, "expanded": molfile, "preferred": molfile},
        "graph": {"skeleton": graph_mol, "expanded": graph_mol, "preferred": graph_mol},
        "detection_graph": graph,
        "label_expansion": {"cxsmiles": cx_result, "expanded": expansion_result},
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
KETCHER_DIR = resolve_ketcher_dir()
app = FastAPI(title="I2BL + Ketcher WebApp")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
TEST_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/examples", StaticFiles(directory=TEST_DIR), name="examples")

HAS_KETCHER = (KETCHER_DIR / "index.html").exists()
if HAS_KETCHER:
    app.mount("/ketcher", StaticFiles(directory=KETCHER_DIR, html=True), name="ketcher")
    if (KETCHER_DIR / "static").exists():
        app.mount("/static", StaticFiles(directory=KETCHER_DIR / "static"), name="ketcher-static")
    if (KETCHER_DIR / "js").exists():
        app.mount("/js", StaticFiles(directory=KETCHER_DIR / "js"), name="ketcher-js")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
async def ketcher_favicon():
    if HAS_KETCHER and (KETCHER_DIR / "favicon.ico").exists():
        return FileResponse(KETCHER_DIR / "favicon.ico")
    raise HTTPException(status_code=404, detail="favicon not found")


@app.get("/manifest.json")
async def ketcher_manifest():
    if HAS_KETCHER and (KETCHER_DIR / "manifest.json").exists():
        return FileResponse(KETCHER_DIR / "manifest.json")
    raise HTTPException(status_code=404, detail="manifest not found")


@app.get("/api/config")
async def api_config():
    """Return webapp configuration."""
    pair_profile = DEFAULT_INFERENCE_PROFILE == "dual_decoder"
    return {
        "model": "shared encoder + dual decoder + pair bond head" if pair_profile else "I2BL_TODO2",
        "inference_profile": DEFAULT_INFERENCE_PROFILE,
        "pair_bundle": str(DEFAULT_PAIR_BUNDLE) if pair_profile else None,
        "ocr_backend": "adapted EasyOCR, 1x/2x/3x/4x" if pair_profile else os.environ.get("I2BL_OCR_BACKEND", "auto"),
        "threshold_controls_active": not pair_profile,
        "config": str(DEFAULT_CONFIG),
        "checkpoint": str(DEFAULT_RESUME),
        "has_ketcher": HAS_KETCHER,
        "ketcher_available": HAS_KETCHER,
        "ketcher_url": "/ketcher/index.html" if HAS_KETCHER else None,
        "ketcher_dir": str(KETCHER_DIR),
        "results_data_dir": str(RESULTS_DATA_DIR),
        "fixed_data_dir": str(FIXED_DATA_DIR),
        "pasted_examples_dir": str(PASTED_EXAMPLES_DIR),
        "training_pairs_dir": str(TRAINING_PAIRS_DIR),
        "default_atom_cutoff": DEFAULT_ATOM_CUTOFF,
        "default_bond_cutoff": DEFAULT_LINE_CUTOFF,
        "default_atom_nms": DEFAULT_ATOM_NMS,
        "default_charge_nms": DEFAULT_CHARGE_NMS,
    }


@app.get("/api/examples")
async def examples():
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    items = []
    if TEST_DIR.exists():
        items.extend(
            {"name": path.name, "url": f"/examples/{path.name}", "source": "test"}
            for path in sorted(TEST_DIR.iterdir())
            if path.is_file() and path.suffix.lower() in allowed_suffixes
        )
    if PASTED_EXAMPLES_DIR.exists():
        items.extend(
            {"name": path.name, "url": f"/uploaded-examples/{path.name}", "source": "pasted"}
            for path in sorted(PASTED_EXAMPLES_DIR.iterdir())
            if path.is_file() and path.suffix.lower() in allowed_suffixes
        )
    unique_items = []
    seen_images = set()
    for item in items:
        if item["source"] != "test":
            unique_items.append(item)
            continue
        image_path = TEST_DIR / item["name"]
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest in seen_images: continue
        seen_images.add(digest)
        unique_items.append(item)
        latest = TRAINING_PAIRS_DIR / digest / "ground_truth.json"
        seed = image_path.with_suffix(".annotation.json")
        legacy = image_path.with_suffix(".json")
        annotation_path = latest if latest.is_file() else seed if seed.is_file() else legacy
        if not annotation_path.is_file(): continue
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        item.update(has_annotation=True, revision=annotation.get("revision", 0),
                    review_status="reviewed" if latest.is_file() else "imported_unreviewed",
                    counts={k:len(annotation.get(k, [])) for k in ("atoms", "charges", "bonds")})
    return {"examples": unique_items}


@app.post("/api/open-example")
async def open_annotated_example(filename: str = Form(...)):
    """Open saved manual labels directly, preferring the newest image-hash revision."""
    image_path = (TEST_DIR / Path(filename).name).resolve()
    if image_path.parent != TEST_DIR or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Example image not found")
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    latest = TRAINING_PAIRS_DIR / digest / "ground_truth.json"
    seed = image_path.with_suffix(".annotation.json")
    legacy = image_path.with_suffix(".json")
    label_path = latest if latest.is_file() else seed if seed.is_file() else legacy
    if not label_path.is_file():
        raise HTTPException(status_code=404, detail="No saved annotation for this example")
    annotation = json.loads(label_path.read_text(encoding="utf-8"))
    if annotation.get("schema") != "i2bl_detection_ground_truth_v1":
        raise HTTPException(status_code=422, detail="Example annotation must use the detection ground-truth schema")
    image = Image.open(image_path).convert("RGB")
    image_size = _image_size_payload(image)
    atoms = list(annotation.get("atoms", [])); charges = list(annotation.get("charges", []))
    bonds = list(annotation.get("bonds", []))
    ac_payload = {"image_size": image_size, "items": atoms + charges}
    b_payload = {"image_size": image_size, "items": bonds}
    rebuilt = _reassemble_detection_payload(ac_payload, b_payload)
    ocr = annotation.get("ocr") or {"items": []}
    result = dict(image=image, atom_items=atoms + charges, bond_items=bonds,
                  skeleton_smiles=rebuilt["smiles"]["skeleton"], expanded_smiles=rebuilt["smiles"]["expanded"],
                  ocr=ocr, ocr_result={}, molfile=rebuilt["molfile"], ketcher_graph=rebuilt["graph"],
                  inference_profile="saved_manual_annotation",
                  annotation_provenance=annotation.get("provenance") or {"source_annotation": str(label_path)},
                  detections={"atoms": _items_to_group(atoms), "charges": _items_to_group(charges),
                              "bonds": _items_to_group(bonds)})
    saved = save_prediction_bundle(result, image_path.name, source="example", source_image_path=image_path)
    saved.update(annotation_revision=annotation.get("revision", 0),
                 annotation_review_status="reviewed" if latest.is_file() else "imported_unreviewed",
                 annotation_saved_at_utc=annotation.get("saved_at_utc"),
                 annotation_items={"ac": atoms + charges, "b": bonds})
    return saved


@app.post("/api/load-history")
async def load_detection_history(result_stem: str = Form(...)):
    save_stem = sanitize_stem(result_stem)
    prediction_path = RESULTS_DATA_DIR / f"{save_stem}_prediction.json"
    if not prediction_path.is_file():
        raise HTTPException(status_code=404, detail=f"Prediction not found: {save_stem}")
    record = json.loads(prediction_path.read_text(encoding="utf-8"))
    image_path = RESULTS_DATA_DIR / record.get("image_file", "")
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Prediction image not found")
    image_sha256 = record.get("image_sha256") or hashlib.sha256(image_path.read_bytes()).hexdigest()
    gt_path = TRAINING_PAIRS_DIR / image_sha256 / "ground_truth.json"
    if not gt_path.is_file():
        return {"found": False, "image_sha256": image_sha256}
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    ac_items = list(ground_truth.get("atoms", [])) + list(ground_truth.get("charges", []))
    return {
        "found": True,
        "image_sha256": image_sha256,
        "revision": ground_truth.get("revision", 1),
        "source_stem": ground_truth.get("source_stem"),
        "legacy_fix_stems": ground_truth.get("legacy_fix_stems", []),
        "ac": {"image_size": ground_truth.get("image", {}).get("size", record.get("image_size", {})), "items": ac_items},
        "b": {"image_size": ground_truth.get("image", {}).get("size", record.get("image_size", {})),
              "items": list(ground_truth.get("bonds", []))},
    }


@app.post("/api/reassemble")
async def reassemble_detections(result_stem: str = Form(...), ac_json: str = Form(...), b_json: str = Form(...)):
    save_stem = sanitize_stem(result_stem)
    prediction_path = RESULTS_DATA_DIR / f"{save_stem}_prediction.json"
    if not prediction_path.is_file():
        raise HTTPException(status_code=404, detail=f"Prediction not found: {save_stem}")
    try:
        ac_payload, b_payload = json.loads(ac_json), json.loads(b_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid detection JSON") from exc
    if not isinstance(ac_payload, dict) or not isinstance(b_payload, dict):
        raise HTTPException(status_code=400, detail="Detection payloads must be objects")
    record = json.loads(prediction_path.read_text(encoding="utf-8"))
    ac_payload.setdefault("image_size", record.get("image_size", {}))
    try:
        rebuilt = _reassemble_detection_payload(ac_payload, b_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not assemble edited detections: {exc}") from exc
    return {"status": "ok", "stem": save_stem, **rebuilt}


@app.get("/uploaded-examples/{filename}")
async def uploaded_example(filename: str):
    # Resolve only direct children so this convenience route cannot traverse paths.
    path = (PASTED_EXAMPLES_DIR / Path(filename).name).resolve()
    if path.parent != PASTED_EXAMPLES_DIR or not path.is_file():
        raise HTTPException(status_code=404, detail="Example image not found")
    return FileResponse(path)


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    source: str = Form("upload"),
    atom_cutoff: float = Form(DEFAULT_ATOM_CUTOFF, ge=0, le=1),
    charge_cutoff: float = Form(DEFAULT_ATOM_CUTOFF, ge=0, le=1),
    bond_cutoff: float = Form(DEFAULT_LINE_CUTOFF, ge=0, le=1),
    atom_nms: float = Form(DEFAULT_ATOM_NMS, ge=0, le=1),
    charge_nms: float = Form(DEFAULT_CHARGE_NMS, ge=0, le=1),
):
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    temp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as handle:
            shutil.copyfileobj(image.file, handle)
            temp_path = handle.name
        try:
            with Image.open(temp_path) as uploaded_image:
                uploaded_image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Upload a valid image file") from exc
        result = _run_i2bl_inference(
            temp_path,
            atom_cutoff=atom_cutoff,
            bond_cutoff=bond_cutoff,
            atom_nms=atom_nms,
            charge_cutoff=charge_cutoff,
            charge_nms=charge_nms,
        )
        if source not in {"clipboard", "upload", "example"}:
            source = "upload"
        saved = save_prediction_bundle(result, image.filename or "upload.png", source=source,
                                       source_image_path=temp_path)
        return saved
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/api/fix")
async def save_fix(
    result_stem: str = Form(...),
    ac_json: str = Form(...),
    b_json: str = Form(...),
    ocr_json: str = Form("{}"),
    molfile_text: str = Form(""),
    graph_json: str = Form("{}"),
    fix_mode: str = Form("detections"),
):
    if fix_mode not in {"detections", "ocr"}:
        raise HTTPException(status_code=400, detail="fix_mode must be detections or ocr")
    try:
        ac_payload = json.loads(ac_json)
        b_payload = json.loads(b_json)
        ocr_payload = json.loads(ocr_json)
        graph_payload = json.loads(graph_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    if not all(isinstance(payload, dict) for payload in (ac_payload, b_payload, ocr_payload, graph_payload)):
        raise HTTPException(status_code=400, detail="JSON payloads must be objects")

    save_stem = sanitize_stem(result_stem)
    prediction_path = RESULTS_DATA_DIR / f"{save_stem}_prediction.json"
    record_dir = FIXED_RECORDS_DIR / save_stem
    ac_path = record_dir / f"{save_stem}_ac.json"
    b_path = record_dir / f"{save_stem}_b.json"
    ocr_path = record_dir / f"{save_stem}_ocr.json"
    molfile_path = record_dir / f"{save_stem}_fixed.mol"
    graph_path = record_dir / f"{save_stem}_fixed_graph.json"

    if not prediction_path.exists():
        raise HTTPException(status_code=404, detail=f"Prediction not found: {save_stem}")

    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        prediction_record = json.loads(prediction_path.read_text(encoding="utf-8"))
        image_file = prediction_record["image_file"]
        record_dir.mkdir(parents=True, exist_ok=True)
        source_image_path = RESULTS_DATA_DIR / image_file
        shutil.copy2(source_image_path, record_dir / "image.png")

        # Save fixed ac/b
        ac_record = {"image_file": image_file, "stem": save_stem, "saved_at_utc": timestamp,
                     "source": "fixed", "data": ac_payload, "ocr_data": ocr_payload}
        b_record = {"image_file": image_file, "stem": save_stem, "saved_at_utc": timestamp,
                     "source": "fixed", "data": b_payload}
        write_json(ac_path, ac_record)
        write_json(b_path, b_record)
        write_json(ocr_path, {"image_file": image_file, "stem": save_stem, "saved_at_utc": timestamp,
                              "source": "fixed", "data": ocr_payload})

        fixed_mol = None
        reassembled_payload = None
        if fix_mode == "detections":
            fixed_graph_payload, fixed_ocr_labels = _graph_from_detection_payload(
                ac_payload, b_payload, prediction_record.get("image_size", {}))
            fixed_mol = _mol_from_i2bl_graph(fixed_graph_payload, fixed_ocr_labels)
            molfile_text = _safe_molblock(fixed_mol)
            graph_payload = fixed_graph_payload
            reassembled_payload = _reassemble_detection_payload(ac_payload, b_payload)
        elif molfile_text:
            molfile_path.write_text(molfile_text, encoding="utf-8")
            try:
                fixed_mol = Chem.MolFromMolBlock(molfile_text, sanitize=False, removeHs=False)
                if fixed_mol is not None:
                    try:
                        Chem.SanitizeMol(fixed_mol, catchErrors=True)
                    except Exception:
                        pass
            except Exception:
                fixed_mol = None

        if molfile_text:
            molfile_path.write_text(molfile_text, encoding="utf-8")
        fixed_graph = _extract_mol_graph(fixed_mol) if fixed_mol is not None else graph_payload
        write_json(graph_path, {
            "fixed_graph": fixed_graph,
            "ketcher_graph_input": graph_payload,
            "saved_at_utc": timestamp,
        })

        # Store the edited boxes and bond bands as a paired, canonical training
        # annotation. Keep original labels too: e.g. Me -> CH3 and '=' -> double.
        image_bytes = source_image_path.read_bytes()
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        pair_dir = TRAINING_PAIRS_DIR / image_sha256
        pair_dir.mkdir(parents=True, exist_ok=True)
        paired_image_path = pair_dir / "image.png"
        if not paired_image_path.exists():
            shutil.copy2(source_image_path, paired_image_path)
        ground_truth_path = pair_dir / "ground_truth.json"
        prior_gt = {}
        if ground_truth_path.exists():
            try:
                prior_gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prior_gt = {}
        # Preserve each Fix click as an immutable training annotation revision.
        # Also archive the current latest file before replacing it, so data
        # written by older app versions is retained on the first new Fix.
        revisions_dir = pair_dir / "revisions"
        revisions_dir.mkdir(parents=True, exist_ok=True)
        prior_revision = int(prior_gt.get("revision", 0))
        if prior_gt and prior_revision > 0:
            prior_stem = sanitize_stem(prior_gt.get("source_stem", save_stem))
            prior_archive = revisions_dir / f"revision_{prior_revision:04d}_{prior_stem}.json"
            if not prior_archive.exists():
                write_json(prior_archive, prior_gt)
        revision = int(prior_gt.get("revision", 0)) + 1
        ground_truth = _canonicalize_detection_ground_truth(
            ac_payload, b_payload, ocr_payload, image_size=prediction_record.get("image_size", {}),
            image_file="image.png", image_sha256=image_sha256, source_stem=save_stem,
            revision=revision, saved_at_utc=timestamp,
        )
        ground_truth["review_status"] = "human_confirmed"
        ground_truth["provenance"] = (prior_gt.get("provenance") or
                                      prediction_record.get("annotation_provenance") or
                                      {"source": "webapp_upload"})
        ground_truth["parent_revision"] = prior_revision
        write_json(ground_truth_path, ground_truth)
        revision_stem = sanitize_stem(save_stem)
        revision_archive = revisions_dir / f"revision_{revision:04d}_{revision_stem}.json"
        write_json(revision_archive, ground_truth)
        _refresh_training_pairs_manifest()

        # Mirror fixed samples into the webapp's selectable test examples.
        # The image hash gives a stable image/label pair name across uploads.
        test_image_name = f"fixed_{image_sha256[:16]}.png"
        test_image_path = TEST_DIR / test_image_name
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        if not test_image_path.exists():
            shutil.copy2(paired_image_path, test_image_path)
        test_label_path = TEST_DIR / f"fixed_{image_sha256[:16]}.json"
        write_json(test_label_path, ground_truth)

        fixed_smiles = ""
        if fixed_mol is not None:
            try:
                fixed_smiles = Chem.MolToSmiles(fixed_mol)
            except Exception:
                fixed_smiles = ""
        fixed_skeleton_smiles = fixed_smiles
        if reassembled_payload:
            fixed_skeleton_smiles = reassembled_payload["smiles"]["skeleton"]
            fixed_smiles = reassembled_payload["smiles"]["expanded"]
        rebuilt = {
            "smiles": {
                "skeleton": fixed_skeleton_smiles if fix_mode == "detections" else prediction_record.get("smiles", {}).get("skeleton", ""),
                "expanded": fixed_smiles or prediction_record.get("smiles", {}).get("expanded", ""),
            },
            "molfile": {
                "skeleton": molfile_text if fix_mode == "detections" else (
                    (RESULTS_DATA_DIR / prediction_record.get("molfile_files", {}).get("skeleton", "")).read_text(encoding="utf-8")
                    if prediction_record.get("molfile_files", {}).get("skeleton") and (RESULTS_DATA_DIR / prediction_record["molfile_files"]["skeleton"]).exists() else ""),
                "expanded": molfile_text,
                "preferred": molfile_text,
            },
            "graph": {
                "skeleton": fixed_graph if fix_mode == "detections" else {},
                "expanded": fixed_graph,
                "preferred": fixed_graph,
            },
        }

        prediction_record["fix_status"] = {
            "fixed_at_utc": timestamp,
            "fixed_data_dir": str(FIXED_DATA_DIR),
            "ac_file": str(ac_path.relative_to(FIXED_DATA_DIR)),
            "b_file": str(b_path.relative_to(FIXED_DATA_DIR)),
            "ocr_file": str(ocr_path.relative_to(FIXED_DATA_DIR)),
            "molfile_file": str(molfile_path.relative_to(FIXED_DATA_DIR)),
            "graph_file": str(graph_path.relative_to(FIXED_DATA_DIR)),
            "fix_mode": fix_mode,
        }
        write_json(prediction_path, prediction_record)
        write_json(record_dir / f"{save_stem}_prediction.json", {
            **prediction_record,
            "source": "fixed",
            "source_prediction": str(prediction_path),
            "smiles": rebuilt["smiles"],
            "ocr": ocr_payload,
            "molfile_files": {"preferred": str(molfile_path.relative_to(FIXED_DATA_DIR))} if molfile_text else {},
            "graph_files": {"preferred": str(graph_path.relative_to(FIXED_DATA_DIR))},
        })
        record_revision_dir = record_dir / "revisions" / f"revision_{revision:04d}"
        record_revision_dir.mkdir(parents=True, exist_ok=True)
        for source_path in (record_dir / "image.png", ac_path, b_path, ocr_path,
                            molfile_path, graph_path, record_dir / f"{save_stem}_prediction.json"):
            if source_path.is_file():
                shutil.copy2(source_path, record_revision_dir / source_path.name)
        write_json(record_revision_dir / "ground_truth.json", ground_truth)
        write_json(record_revision_dir / "fix_metadata.json", {
            "result_stem": save_stem,
            "image_sha256": image_sha256,
            "revision": revision,
            "saved_at_utc": timestamp,
            "fix_mode": fix_mode,
        })
        _refresh_fixed_records_manifest()

        return {"status": "ok", "ok": True, "stem": save_stem, "saved_at_utc": timestamp,
                "fix_mode": fix_mode,
                "rebuilt": rebuilt, "training_pair": {
                    "image": str(paired_image_path), "ground_truth": str(ground_truth_path),
                    "image_sha256": image_sha256, "revision": revision,
                }, "test_example": {
                    "image": str(test_image_path), "ground_truth": str(test_label_path),
                    "example_name": test_image_name,
                }, "storage": {"fixed_data_dir": str(FIXED_DATA_DIR),
                               "test_dir": str(TEST_DIR)}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2861)
