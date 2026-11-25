"""Experimental appearance-scale/topology-aware charge assignment.

No source SMILES, GT owners or atom formal_charge attributes are consulted.
This stays opt-in until full-corpus regression is checked.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import linear_sum_assignment
from .charge_association import ATOM_NAMES,CHARGES,ORDER,valence_valid

@lru_cache(maxsize=4096)
def implicit_hydrogens(symbol,orders,value):
    if symbol not in ATOM_NAMES[1:] or any(v==1.5 for v in orders):return None
    from rdkit import Chem,rdBase
    try:
        with rdBase.BlockLogs():
            mol=Chem.RWMol();atom=Chem.Atom(symbol);atom.SetFormalCharge(value);mol.AddAtom(atom)
            for order in orders:
                mol.AddAtom(Chem.Atom(0));mol.AddBond(0,mol.GetNumAtoms()-1,{1.:Chem.BondType.SINGLE,2.:Chem.BondType.DOUBLE,3.:Chem.BondType.TRIPLE}[order])
            mol.UpdatePropertyCache(strict=True)
            return int(mol.GetAtomWithIdx(0).GetNumImplicitHs())
    except (ValueError,RuntimeError):return None

def assign_adaptive(atoms,charges,bonds,lines,guarded=False,preserve_positive_hetero=False):
    if not charges:return []
    from .charge_association import assign_charges
    baseline=assign_charges(atoms,charges,bonds)
    if not atoms:return baseline
    boxes=np.asarray([a['bbox'] for a in atoms]);centers=boxes[:,:2]+boxes[:,2:]/2
    qb=np.asarray([q['bbox'] for q in charges]);points=qb[:,:2]+qb[:,2:]/2
    geo=np.linalg.norm(np.maximum(np.maximum(boxes[None,:,:2]-points[:,None],points[:,None]-(boxes[:,:2]+boxes[:,2:])[None]),0),axis=-1)+.02*np.linalg.norm(centers[None]-points[:,None],axis=-1)
    nearest=geo.argmin(1);orders=[[] for _ in atoms]
    for b in bonds:
        o=ORDER[b['type']];orders[b['begin']].append(o);orders[b['end']].append(o)
    orders=[tuple(sorted(v)) for v in orders]
    symbols=[a.get('symbol',ATOM_NAMES[a.get('label',a.get('category_id',0))]) for a in atoms]
    values=[int(q['value']) if 'value' in q else CHARGES[q['label']] for q in charges]
    line_array=np.asarray(lines,float).reshape(-1,2,2)
    lengths=np.linalg.norm(line_array[:,1]-line_array[:,0],axis=1)
    typical=float(np.median(lengths)) if len(lengths) else 12.
    radius=np.clip(np.maximum(1.5*np.linalg.norm(qb[:,2:],axis=1),.75*typical),6,40)
    n=len(atoms);count=len(charges);cost=np.full((count,n+count),1e8);features={}
    for qi,value in enumerate(values):
        for ai in np.flatnonzero(geo[qi]<=geo[qi,nearest[qi]]+radius[qi]):
            symbol=symbols[ai];os=orders[ai];neutral=valence_valid(symbol,os,0);charged=valence_valid(symbol,os,value)
            hs=implicit_hydrogens(symbol,os,value);nhs=implicit_hydrogens(symbol,os,0)
            # Some source annotations include nonstandard three-bond N anions.
            # Without radical/H-state predictions, strict RDKit rejection is not
            # reliable evidence against their ownership; retain geometric prior.
            uncertain_n=(symbol=='N' and value==-1 and charged is False and sum(os)==3)
            penalty=radius[qi] if charged is False and not uncertain_n else 0.
            hydrogen=.5*radius[qi]*min(2,hs) if hs is not None else 0.
            needed=radius[qi] if neutral is False and charged is True else 0.
            onium=.5*radius[qi] if value>0 and symbol in ['N','O','P','S'] and hs==0 and nhs is not None and nhs>0 else 0.
            cost[qi,ai]=geo[qi,ai]+penalty+hydrogen-needed-onium
            features[qi,int(ai)]=dict(neutral_valid=neutral,charged_valid=charged,implicit_h=hs,uncertain_n=uncertain_n)
        cost[qi,n+qi]=geo[qi,nearest[qi]]+3*radius[qi]
    rr,cc=linear_sum_assignment(cost);out=[None]*count
    for qi,ai in zip(rr,cc):
        owner=int(ai) if ai<n else None;old=baseline[qi]['atom_index']
        accepted=True
        if guarded and owner!=old:
            oldcost=cost[qi,old] if old is not None else cost[qi,n+qi]
            gain=oldcost-cost[qi,ai];accepted=bool(gain>=.5*radius[qi])
            if not accepted:owner=old
        # N/O/P/S cations can carry implicit hydrogens not represented in the
        # atom-class input. Do not move an already plausible heteroatom cation
        # onto carbon merely because the local implicit-H prior favors it.
        if preserve_positive_hetero and owner is not None and old is not None and owner!=old:
            if values[qi]>0 and symbols[old] in ['N','O','P','S'] and symbols[owner]=='C' and valence_valid(symbols[old],orders[old],values[qi]) is not False:
                owner=old;accepted=False
        out[qi]=dict(charge_index=int(qi),atom_index=owner,baseline_atom=old,radius=float(radius[qi]),
                     accepted=accepted,features=features.get((int(qi),int(owner))) if owner is not None else None)
    # A guarded combination must not reintroduce a many-charge/same-atom match.
    owners=[x['atom_index'] for x in out if x['atom_index'] is not None]
    if len(set(owners))!=len(owners):return baseline
    return out
