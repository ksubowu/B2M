"""Collapse split/textual heavy-atom–H notations into heavy-atom vertices."""
from copy import deepcopy
import re


_SUBSCRIPT_TRANSLATION = str.maketrans({"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4"})
_ELEMENTS = frozenset(
    "B C N O F Si P S Cl Br I Se As Te At Po Li Be Na Mg Al K Ca Fe Co Ni Cu Zn Ag Au Hg".split()
)
_ELEMENT_CLASS_IDS = {"C": 1, "O": 2, "N": 3, "Cl": 4, "Br": 5, "S": 6,
                     "F": 7, "B": 8, "I": 9, "P": 10, "H": 11, "Si": 12}


def _notation_formula(text):
    # Preserve atom-symbol case: lowercase n in aromatic nH is chemically
    # meaningful and must not be rewritten as an aliphatic N-H label.
    value = re.sub(r"\s+", "", str(text or "")).translate(_SUBSCRIPT_TRANSLATION)
    element_pattern = r"([A-Z][a-z]?)"
    match = re.fullmatch(element_pattern + r"H([1-9]?)", value)
    if match:
        if match.group(1) in _ELEMENTS:
            return match.group(1), int(match.group(2) or 1), value
    match = re.fullmatch(r"H([1-9]?)" + element_pattern, value)
    if match:
        if match.group(2) in _ELEMENTS:
            return match.group(2), int(match.group(1) or 1), value
    return None


def _close_text_boxes(box_a, box_b):
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return False
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    aw, ah = max(1.0, ax2 - ax1), max(1.0, ay2 - ay1)
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    x_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    y_overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
    x_gap = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    y_gap = max(0.0, max(ay1, by1) - min(ay2, by2))
    min_h, min_w = min(ah, bh), min(aw, bw)
    inline = y_overlap / min_h >= 0.45 and x_gap <= max(2.5, 0.28 * min_h)
    stacked = x_overlap / min_w >= 0.45 and y_gap <= max(2.5, 0.28 * min_w)
    return inline or stacked


def collapse_hydrogen_notation(graph, text_labels=None):
    """Merge adjacent H detections or recognized CHx/NHx text into one C/N atom.

    The detector view is preserved elsewhere. This changes only the molecular
    graph used for assembly and records the original atom notation on the atom.
    """
    graph = deepcopy(graph)
    text_labels = {int(k): dict(v) for k, v in (text_labels or {}).items()}
    atoms = graph.get("atoms", [])
    by_index = {int(a["index"]): a for a in atoms}
    drop_to_target = {}
    audit = []

    # A complete recognized label such as OH, NH₂ or H₃C directly specifies
    # one heavy atom; final hydrogen valence is inferred from bonds and charge.
    for atom in atoms:
        index = int(atom["index"])
        text = text_labels.get(index, {}).get("raw_text") or text_labels.get(index, {}).get("text")
        formula = _notation_formula(text)
        if formula:
            symbol, hydrogen_count, notation = formula
            atom["symbol"] = atom["label_name"] = symbol
            atom["atom_notation"] = str(text).strip()
            atom["label_id"] = _ELEMENT_CLASS_IDS.get(symbol, atom.get("label_id", 0))
            text_labels.pop(index, None)
            audit.append({"action": "decode_composite_label", "atom": index,
                          "notation": notation, "element": symbol,
                          "hydrogens_in_notation": hydrogen_count,
                          "valence_source": "bonds_and_formal_charge"})

    # A model may split a tightly typeset N H / C H into two atom boxes.
    heavy_atoms = [a for a in atoms if a.get("symbol") in _ELEMENTS]
    hydrogen_atoms = [a for a in atoms if a.get("symbol") == "H"]
    for hydrogen in hydrogen_atoms:
        candidates = [a for a in heavy_atoms if _close_text_boxes(a.get("box"), hydrogen.get("box"))]
        if not candidates:
            continue
        target = min(candidates, key=lambda a: (
            (float(a["center"][0]) - float(hydrogen["center"][0])) ** 2
            + (float(a["center"][1]) - float(hydrogen["center"][1])) ** 2,
            int(a["index"]),
        ))
        target_index, hydrogen_index = int(target["index"]), int(hydrogen["index"])
        drop_to_target[hydrogen_index] = target_index
        target["formal_charge"] = int(target.get("formal_charge", 0)) + int(hydrogen.get("formal_charge", 0))
        target.setdefault("charge_sources", []).extend(hydrogen.get("charge_sources", []))
        # Keep the heavy-atom anchor for bond attachment, but expand its editor box
        # to cover the complete NH/CH notation.
        x1, y1, x2, y2 = target["box"]
        hx1, hy1, hx2, hy2 = hydrogen["box"]
        target["box"] = [min(x1, hx1), min(y1, hy1), max(x2, hx2), max(y2, hy2)]
        notation = target.get("atom_notation") or target["symbol"]
        target["atom_notation"] = notation + "H"
        audit.append({"action": "merge_adjacent_h_box", "hydrogen_atom": hydrogen_index,
                      "heavy_atom": target_index, "notation": target["atom_notation"],
                      "valence_source": "bonds_and_formal_charge"})

    if not audit:
        return graph, text_labels, audit

    # Reindex vertices and reconnect any real bonds that terminate on a merged H.
    retained = [a for a in atoms if int(a["index"]) not in drop_to_target]
    old_to_new = {int(a["index"]): idx for idx, a in enumerate(retained)}
    remapped = []
    for atom in retained:
        old = int(atom["index"])
        atom["index"] = old_to_new[old]
        atom["bond_indices"] = []
        remapped.append(atom)
    kept_bonds = []
    for bond in graph.get("bonds", []):
        begin = int(bond["begin_atom_idx"])
        end = int(bond["end_atom_idx"])
        begin = drop_to_target.get(begin, begin)
        end = drop_to_target.get(end, end)
        if begin == end or begin not in old_to_new or end not in old_to_new:
            continue
        bond = dict(bond)
        bond["index"] = len(kept_bonds)
        bond["begin_atom_idx"] = old_to_new[begin]
        bond["end_atom_idx"] = old_to_new[end]
        bond["atom_pair"] = sorted((bond["begin_atom_idx"], bond["end_atom_idx"]))
        remapped[ bond["begin_atom_idx"] ]["bond_indices"].append(bond["index"])
        remapped[ bond["end_atom_idx"] ]["bond_indices"].append(bond["index"])
        kept_bonds.append(bond)

    new_labels = {}
    for old_index, label in text_labels.items():
        mapped = drop_to_target.get(old_index, old_index)
        if mapped in old_to_new:
            new_labels[old_to_new[mapped]] = label
    graph["atoms"], graph["bonds"] = remapped, kept_bonds
    graph.setdefault("mappings", {})["atom_to_bonds"] = {
        atom["index"]: list(atom["bond_indices"]) for atom in remapped
    }
    counts = graph.setdefault("counts", {})
    counts["kept_atoms"] = len(remapped)
    counts["kept_bonds"] = len(kept_bonds)
    graph["hydrogen_notation"] = audit
    return graph, new_labels, audit
