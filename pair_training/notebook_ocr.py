"""Inspectable multi-scale OCR and conservative, stereo-preserving label expansion.

This notebook path does not change production WebApp recognition policy.
"""
import math

import numpy as np
from PIL import Image, ImageOps
from rdkit import Chem
from rdkit.Chem import rdDepictor


def _chem_helpers():
    from engine.rtv4.chem_ocr_expand import (ABBREVIATIONS, ELEMENTS, RGROUP_SYMBOLS,
                                            normalize_ocr_text, expand_abbreviation)
    return ABBREVIATIONS, ELEMENTS, RGROUP_SYMBOLS, normalize_ocr_text, expand_abbreviation


def paddle_recognizer():
    from engine.rtv4.chem_ocr_expand import _get_paddleocr_instances, _paddleocr_recognize
    primary, _ = _get_paddleocr_instances(use_gpu=False)
    if primary is None:
        raise RuntimeError('PaddleOCR不可用；检查bl2m环境及本地OCR模型，不会静默跳过OCR')
    return lambda rgb: _paddleocr_recognize(primary, rgb)


def complete_text_box(image, box, forbidden_centers=()):
    """Extend a clipped label box to neighboring glyph components, not bond ink.

    Thin bond strokes and large molecule components are excluded. Never absorb
    a component containing another predicted atom center. This only changes OCR
    crops; atom detection, graph indices and bonds are never merged or moved.
    """
    from scipy import ndimage
    rgb=np.asarray(image.convert('RGB'))
    ink=rgb.min(axis=2)<180
    labeled,_=ndimage.label(ink,structure=np.ones((3,3),dtype=np.uint8))
    stats=[]
    for index,region in enumerate(ndimage.find_objects(labeled),1):
        if region is None:continue
        ys,xs=region
        stats.append((xs.start,ys.start,xs.stop-xs.start,ys.stop-ys.start,
                      int(np.count_nonzero(labeled[region]==index))))
    x1,y1,x2,y2=map(float,box)
    reference_height=max(3.,y2-y1)
    components=[]
    for x,y,w,h,area in stats:
        if area<2 or h<reference_height*.25 or h>reference_height*1.8 or w>reference_height*4:
            continue
        if any(x-1<=cx<=x+w+1 and y-1<=cy<=y+h+1 for cx,cy in forbidden_centers):
            continue
        components.append((int(x),int(y),int(x+w),int(y+h)))
    expanded=[x1,y1,x2,y2]
    for _ in range(len(components)):
        changed=False
        for a,b,c,d in components:
            horizontal_gap=max(a-expanded[2],expanded[0]-c,0)
            overlap=max(0,min(d,expanded[3])-max(b,expanded[1]))
            if horizontal_gap<=reference_height*.5 and overlap>=min(d-b,reference_height)*.4:
                proposed=[min(expanded[0],a),min(expanded[1],b),max(expanded[2],c),max(expanded[3],d)]
                if proposed!=expanded and proposed[2]-proposed[0]<=reference_height*12:
                    # Avoid encompassing a separate known graph atom, even if its
                    # glyph was not itself the newly added component.
                    if any(proposed[0]<=cx<=proposed[2] and proposed[1]<=cy<=proposed[3]
                           and not (x1<=cx<=x2 and y1<=cy<=y2) for cx,cy in forbidden_centers):
                        continue
                    expanded=proposed;changed=True
        if not changed:break
    return expanded


def group_text_targets(graph, image):
    """Group split unknown text only when extra detections are isolated.

    Never merge two bonded nodes, two nodes with independent connections, or
    boxes separated by a near-solid black bond stroke. Original graph is intact.
    """
    targets=[dict(a, merged_atom_indices=[a['index']]) for a in graph['atoms'] if a['symbol']=='*']
    if not targets:return []
    degree={a['index']:0 for a in graph['atoms']}
    for b in graph.get('bonds',[]):
        degree[b['begin_atom_idx']]+=1;degree[b['end_atom_idx']]+=1
    ink=np.asarray(image.convert('RGB')).min(axis=2)<180
    changed=True
    while changed:
        changed=False
        for i,a in enumerate(targets):
            if changed:break
            for j in range(i+1,len(targets)):
                b=targets[j]
                if degree[a['index']]>0 and degree[b['index']]>0:continue
                x,y,u,v=a['box']; xx,yy,uu,vv=b['box']
                h=min(v-y,vv-yy)
                overlap=min(v,vv)-max(y,yy)
                gap=max(xx-u,x-uu,0)
                if h<=0 or overlap<h*.5 or gap>h*.7:continue
                union=[min(x,xx),min(y,yy),max(u,uu),max(v,vv)]
                member_ids=a['merged_atom_indices']+b['merged_atom_indices']
                if union[2]-union[0]>max(v-y,vv-yy)*8:continue
                if any(c['index'] not in member_ids and union[0]<=c['center'][0]<=union[2]
                       and union[1]<=c['center'][1]<=union[3] for c in graph['atoms'] if 'center' in c):continue
                if gap>=3:
                    left,right=(a,b) if x<xx else (b,a)
                    l=math.ceil(left['box'][2]);r=math.floor(right['box'][0])
                    top=max(0,math.floor(max(y,yy)));bottom=min(image.height,math.ceil(min(v,vv)))
                    strip=ink[top:bottom,max(0,l):min(image.width,r)]
                    if strip.size and strip.shape[1]>=3:
                        thin_stroke=strip.any(axis=1).sum()<=max(3,h*.45)
                        if thin_stroke and (strip.mean(axis=1)>=.8).any():continue
                keeper=a if degree[a['index']]>=degree[b['index']] else b
                targets[i]=dict(keeper,box=union,merged_atom_indices=member_ids)
                targets.pop(j);changed=True;break
    return targets


def recognize_multiscale(graph, image, scales=(1., 2., 3., 4.), margins=(2, 4),
                         min_confidence=.5, recognizer=None):
    """Select the highest finite nonempty OCR score, with no dictionary override.

    Each scale enlarges a crop from the original image. Scores across scales
    come from the same backend; they are ranking scores, not calibrated accuracy.
    Low-confidence or unmappable text remains visible but is not auto-expanded.
    """
    if not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('OCR scales must be positive finite values')
    if not margins or any(not math.isfinite(m) or m < 0 for m in margins):
        raise ValueError('OCR margins must be nonnegative finite values')
    if not 0 <= min_confidence <= 1:
        raise ValueError('OCR min_confidence must lie in [0,1]')
    targets = group_text_targets(graph, image)
    if not targets:
        return {}, []  # Do not load OCR when there are no unknown atoms.
    recognizer = recognizer or paddle_recognizer()
    abbrevs, elements, rgroups, normalize, expand = _chem_helpers()
    labels, records = {}, []
    for atom in targets:
        candidates = []
        others=[a['center'] for a in graph['atoms'] if a['index'] not in atom['merged_atom_indices'] and 'center' in a]
        recognition_box=complete_text_box(image,atom['box'],others)
        for margin in margins:
            x1,y1,x2,y2 = recognition_box
            box = (max(0,math.floor(x1-margin)), max(0,math.floor(y1-margin)),
                   min(image.width,math.ceil(x2+margin)), min(image.height,math.ceil(y2+margin)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop = image.convert('RGB').crop(box)
            for scale in scales:
                resized = crop.resize((max(2,round(crop.width*scale)),max(2,round(crop.height*scale))),
                                      Image.Resampling.LANCZOS)
                padded = ImageOps.expand(resized, border=4, fill='white')
                try:
                    raw, score = recognizer(np.asarray(padded))
                    score = float(score)
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError('nonfinite/out-of-range OCR confidence')
                    raw = str(raw or '').strip()
                    # Preserve genuine R/* labels instead of rewriting them away.
                    text = raw if raw in rgroups or raw in abbrevs or raw in elements or raw == '*' else normalize(raw)
                    error = None
                except Exception as exc:
                    raw, text, score, error = '', '', 0., f'{type(exc).__name__}: {exc}'
                candidates.append(dict(atom_index=atom['index'], prediction_index=atom.get('prediction_index'),
                                       scale=float(scale), margin=float(margin), crop_box=list(box),
                                       raw_text=raw, text=text, score=score, error=error, selected=False,
                                       original_atom_box=list(atom['box']), recognition_box=recognition_box))
        nonempty = [c for c in candidates if c['raw_text'] and not c['error']]
        if nonempty:
            best = max(nonempty, key=lambda c:c['score'])
            best['selected'] = True
            text = best['text']
            supported = text in abbrevs or (text in elements and text not in rgroups)
            label_type = ('abbreviation' if text in abbrevs else 'rgroup' if text in rgroups or text == '*'
                          else 'element' if text in elements else 'unmapped')
            labels[atom['index']] = dict(text=text, raw_text=best['raw_text'], score=best['score'],
                scale=best['scale'], margin=best['margin'], box=best['crop_box'],
                label_type=label_type, smiles=expand(text) if supported else '[*]',
                eligible_for_expansion=supported and best['score'] >= min_confidence,
                selection='highest_raw_ocr_confidence', min_confidence=min_confidence,
                merged_atom_indices=atom['merged_atom_indices'])
        records.extend(candidates)
    return labels, records


def expand_labels(mol, graph, labels):
    """Replace wildcard root atoms in place; do not reorder existing atom indices.

    Keep original chiral tags and bonds. Only dictionary abbreviations on a
    terminal attachment or element labels are applied. Each replacement is
    sanitized independently; failed replacements retain their original wildcard.
    R groups remain wildcards carrying their literal label, not invented atoms.
    """
    if mol is None:
        return None, dict(valid=False, smiles=None, error='No input molecule', applied=[], retained=[])
    abbrevs, elements, rgroups, _, _ = _chem_helpers()
    current = Chem.Mol(mol)
    by_graph = {a['index']:i for i,a in enumerate(graph['atoms'])}
    applied, retained = [], []
    isolated_to_remove=set()
    for index, label in labels.items():
        idx = by_graph[int(index)]
        text = label['text']
        if current.GetAtomWithIdx(idx).GetAtomicNum() != 0:
            retained.append(dict(atom_index=index,text=text,reason='not_a_wildcard')); continue
        # Do not pass arbitrary OCR punctuation to RDKit drawing markup:
        # an atomLabel consisting of ':' can segfault the native renderer.
        # Unsupported raw text is retained in JSON/tables instead.
        if text in rgroups or text == '*':
            current.GetAtomWithIdx(idx).SetProp('atomLabel',text)
        if not label['eligible_for_expansion']:
            retained.append(dict(atom_index=index,text=text,reason='low_confidence_or_unmapped_or_rgroup')); continue
        trial = Chem.RWMol(current)
        try:
            if text in elements and text not in abbrevs:
                replacement = Chem.Atom(text)
                replacement.SetFormalCharge(current.GetAtomWithIdx(idx).GetFormalCharge())
                trial.ReplaceAtom(idx, replacement)
            else:
                if trial.GetAtomWithIdx(idx).GetDegree() != 1:
                    raise ValueError('缩写需要单一外部连接，避免猜测多连接位点')
                fragment = Chem.MolFromSmiles(label['smiles'])
                if fragment is None or fragment.GetNumAtoms() == 0:
                    raise ValueError('invalid abbreviation SMILES')
                # Dictionary attachment root is atom0. Only a single fragment is accepted.
                if len(Chem.GetMolFrags(fragment)) != 1:
                    raise ValueError('disconnected abbreviation')
                root = Chem.Atom(fragment.GetAtomWithIdx(0)); root.SetNumRadicalElectrons(0)
                trial.ReplaceAtom(idx,root)
                mapping = {0:idx}
                for a in list(fragment.GetAtoms())[1:]:
                    mapping[a.GetIdx()] = trial.AddAtom(Chem.Atom(a))
                for b in fragment.GetBonds():
                    trial.AddBond(mapping[b.GetBeginAtomIdx()],mapping[b.GetEndAtomIdx()],b.GetBondType())
            candidate = trial.GetMol()
            Chem.SanitizeMol(candidate)
            current = candidate
            members=label.get('merged_atom_indices',[index])
            removed=[]
            for member in members:
                other=by_graph[int(member)]
                if other!=idx and current.GetAtomWithIdx(other).GetDegree()==0 and current.GetAtomWithIdx(other).GetAtomicNum()==0:
                    isolated_to_remove.add(other);removed.append(member)
            applied.append(dict(atom_index=index,text=text,score=label['score'],smiles=label['smiles'],
                                merged_isolated_atom_indices=removed))
        except Exception as exc:
            retained.append(dict(atom_index=index,text=text,reason=f'{type(exc).__name__}: {exc}'))
    if isolated_to_remove:
        working=Chem.RWMol(current)
        for index in sorted(isolated_to_remove,reverse=True):working.RemoveAtom(index)
        current=working.GetMol()
    try:
        Chem.SanitizeMol(current)
        # Existing stereocenters retain their chiral tags across in-place grafts.
        Chem.AssignStereochemistry(current, cleanIt=True, force=True)
        smiles = Chem.MolToSmiles(current,isomericSmiles=True)
        if applied:
            # Old wedges encode the old 2D geometry. Rebuild them from the
            # retained chiral tags after changing coordinates, not vice versa.
            for bond in current.GetBonds():
                if bond.GetBondDir() in (Chem.BondDir.BEGINWEDGE,Chem.BondDir.BEGINDASH):
                    bond.SetBondDir(Chem.BondDir.NONE)
            rdDepictor.Compute2DCoords(current)
            Chem.WedgeMolBonds(current,current.GetConformer())
        return current, dict(valid=True,smiles=smiles,error=None,applied=applied,retained=retained,
                             remaining_wildcards=sum(a.GetAtomicNum()==0 for a in current.GetAtoms()))
    except Exception as exc:
        return current,dict(valid=False,smiles=None,error=f'{type(exc).__name__}: {exc}',applied=applied,retained=retained)


def labeled_cxsmiles(mol, graph, labels):
    """Serialize OCR literal strings on wildcard atoms without expanding them.

    Atom-label indexing is assigned by RDKit during canonical SMILES writing;
    never construct the $...$ suffix by manually guessing SMILES atom order.
    The returned molecule is for serialization, not rendering arbitrary OCR
    markup (e.g. ':' is unsafe in the native RDKit image renderer).
    """
    if mol is None:return None,dict(valid=False,cxsmiles=None,error='No input molecule')
    current=Chem.RWMol(mol)
    by_graph={a['index']:i for i,a in enumerate(graph['atoms'])}
    recorded=[];remove=set()
    for graph_index,label in labels.items():
        idx=by_graph[int(graph_index)]
        raw=str(label.get('raw_text',label.get('text','')))
        if not raw:continue
        current.GetAtomWithIdx(idx).SetProp('atomLabel',raw)
        recorded.append(dict(graph_atom_index=int(graph_index),raw_text=raw,score=label.get('score'),
                             normalized_text=label.get('text')))
        if current.GetAtomWithIdx(idx).GetAtomicNum()==0 and label.get('score',1.)>=label.get('min_confidence',.5):
            for member in label.get('merged_atom_indices',[]):
                other=by_graph[int(member)]
                if other!=idx and current.GetAtomWithIdx(other).GetDegree()==0 and current.GetAtomWithIdx(other).GetAtomicNum()==0:
                    remove.add(other)
    for idx in sorted(remove,reverse=True):current.RemoveAtom(idx)
    labeled=current.GetMol()
    try:
        Chem.SanitizeMol(labeled)
        params=Chem.SmilesWriteParams();params.canonical=True;params.doIsomericSmiles=True
        cx=Chem.MolToCXSmiles(labeled,params,Chem.CXSmilesFields.CX_ATOM_LABELS)
        restored=Chem.MolFromSmiles(cx)
        if restored is None:raise ValueError('CXSMILES could not be parsed')
        recovered=sorted(a.GetProp('atomLabel') for a in restored.GetAtoms() if a.HasProp('atomLabel'))
        intended=sorted(a.GetProp('atomLabel') for a in labeled.GetAtoms() if a.HasProp('atomLabel'))
        if recovered!=intended:raise ValueError('CXSMILES atom labels changed on roundtrip')
        return labeled,dict(valid=True,cxsmiles=cx,error=None,labels=recorded,
                            atoms=labeled.GetNumAtoms(),bonds=labeled.GetNumBonds(),
                            merged_isolated_atom_indices=sorted(remove),expanded=False)
    except Exception as exc:
        return labeled,dict(valid=False,cxsmiles=None,error=f'{type(exc).__name__}: {exc}',labels=recorded)
