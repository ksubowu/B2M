"""Assign charge detections using local geometry, exclusive assignment and valence.

Inputs contain predicted symbols, boxes, charge values and assembled bonds.
GT formal charges/owners/IDs are never read. Unknown labels and aromatic local
stars fall back to geometry rather than imposing unsupported valence rules.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import linear_sum_assignment

ATOM_NAMES=['other','C','O','N','Cl','Br','S','F','B','I','P','H','Si']
CHARGES={13:-4,14:-2,15:-1,16:1,17:2}
ORDER={'single':1.,'double':2.,'triple':3.,'aromatic':1.5,'wedge':1.,'dash':1.}


@lru_cache(maxsize=4096)
def valence_valid(symbol,orders,value):
    if symbol not in ATOM_NAMES[1:] or any(v==1.5 for v in orders):return None
    from rdkit import Chem,rdBase
    try:
        with rdBase.BlockLogs():
            mol=Chem.RWMol();center=Chem.Atom(symbol);center.SetFormalCharge(int(value));mol.AddAtom(center)
            for order in orders:
                dummy=Chem.Atom(0);dummy.SetNoImplicit(True);mol.AddAtom(dummy)
                mol.AddBond(0,mol.GetNumAtoms()-1,{1.:Chem.BondType.SINGLE,2.:Chem.BondType.DOUBLE,3.:Chem.BondType.TRIPLE}[order])
            mol.UpdatePropertyCache(strict=True)
        return True
    except (ValueError,RuntimeError):return False


def assign_charges(atoms,charges,bonds=(),method='joint_valence',lines=None):
    """Return one diagnostic assignment per charge. Boxes use xywh pixels.

    Geometry: point-to-rectangle + .02 center distance (legacy score).
    Joint mode: at most one charge annotation per atom, a SMALL local ambiguity
    window, SOFT valence penalty, and preference for resolving an existing local
    valence violation. Nonstandard source chemistry must not be hard-rejected.
    A private dummy column allows unassigned duplicate/false detections.
    This is a local valence consistency check, not full molecule sanitization.
    """
    if method=='safe_adaptive':
        if lines is None:raise ValueError('safe_adaptive requires paired line coordinates for its scale estimate')
        from .charge_adaptive import assign_adaptive
        return assign_adaptive(atoms,charges,bonds,lines,preserve_positive_hetero=True)
    if method not in ['legacy','joint_valence']:raise ValueError(method)
    if not charges:return []
    if not atoms:return [dict(charge_index=i,atom_index=None,reason='no_atoms') for i in range(len(charges))]
    boxes=np.asarray([a['bbox'] for a in atoms],float).reshape(-1,4)
    centers=boxes[:,:2]+boxes[:,2:]/2
    qboxes=np.asarray([q['bbox'] for q in charges],float).reshape(-1,4)
    points=qboxes[:,:2]+qboxes[:,2:]/2
    distances=np.linalg.norm(np.maximum(np.maximum(boxes[None,:,:2]-points[:,None],points[:,None]-(boxes[:,:2]+boxes[:,2:])[None]),0),axis=-1)
    geometry=distances+.02*np.linalg.norm(points[:,None]-centers[None],axis=-1)
    nearest=geometry.argmin(1)
    if method=='legacy':return [dict(charge_index=i,atom_index=int(j),legacy_atom_index=int(j),geometry_score=float(geometry[i,j]),reason='nearest_box') for i,j in enumerate(nearest)]
    orders=[[] for _ in atoms]
    for bond in bonds:
        a,b=int(bond['begin']),int(bond['end'])
        if a==b or min(a,b)<0 or max(a,b)>=len(atoms):continue
        name=bond.get('type')
        if name is None:name=['single','double','triple','aromatic','wedge','dash'][int(bond['label'])]
        order=ORDER.get(name)
        if order is not None:orders[a].append(order);orders[b].append(order)
    orders=[tuple(sorted(v)) for v in orders]
    symbols=[]
    for atom in atoms:
        label=atom.get('label',atom.get('category_id',0))
        symbols.append(atom.get('symbol',ATOM_NAMES[label] if 0<=label<len(ATOM_NAMES) else 'other'))
    values=[int(q['value']) if 'value' in q else CHARGES[int(q['label'])] for q in charges]
    # All production coordinates are 640px letterbox. Chemistry only breaks
    # close geometric ties (2..6px), never overrides a clearly nearer atom.
    radius=np.clip(.25*np.linalg.norm(qboxes[:,2:],axis=1),2.,6.)
    count=len(charges);n=len(atoms);cost=np.full((count,n+count),1e8)
    validity={};neutral=[valence_valid(s,o,0) for s,o in zip(symbols,orders)]
    for qi,value in enumerate(values):
        local=geometry[qi]<=geometry[qi,nearest[qi]]+radius[qi]
        for ai in np.flatnonzero(local):
            valid=valence_valid(symbols[ai],orders[ai],value);validity[qi,int(ai)]=valid
            penalty=radius[qi] if valid is False else 0.
            resolves=radius[qi]*.5 if neutral[ai] is False and valid is True else 0.
            cost[qi,ai]=geometry[qi,ai]+penalty-resolves
        cost[qi,n+qi]=geometry[qi,nearest[qi]]+3*radius[qi]
    rows,columns=linear_sum_assignment(cost);output=[None]*count
    for qi,ai in zip(rows,columns):
        owner=int(ai) if ai<n else None
        output[qi]=dict(charge_index=int(qi),atom_index=owner,legacy_atom_index=int(nearest[qi]),
            geometry_score=float(geometry[qi,ai]) if owner is not None else None,
            charged_valence_valid=validity.get((int(qi),int(ai))),
            neutral_valence_valid=neutral[ai] if owner is not None else None,
            reason='joint_local_geometry_and_valence' if owner is not None else 'no_available_compatible_local_atom')
    return output
