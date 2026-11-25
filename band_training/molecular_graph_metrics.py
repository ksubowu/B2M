"""Exact dummy/label graph comparison and reference-only alias folding.

No predicted geometry or OCR text is used to modify the reference graph.
MCS is a bounded diagnostic only; it never turns an incorrect graph into a hit.
"""
import itertools,unicodedata
from collections import Counter,defaultdict
import networkx as nx
from .mixed_data import ATOM_CLASSES


def normalize_text(value):
    # Format normalization only. Preserve case, punctuation and numeric suffixes.
    return ''.join(unicodedata.normalize('NFKC',str(value)).split())


def simple_exact(a,b):
    if len(a)!=len(b) or a.number_of_edges()!=b.number_of_edges():return None
    match=nx.algorithms.isomorphism.GraphMatcher(a,b,
        node_match=lambda x,y:(x['symbol'],x['charge'],x.get('opaque',False))==(y['symbol'],y['charge'],y.get('opaque',False)),
        edge_match=lambda x,y:x['label']==y['label'])
    return next(match.isomorphisms_iter(),None)


def native_graph(molecule,aromatize=True):
    mol=molecule.clone()
    if aromatize:
        try:mol.aromatize()
        except Exception:pass
    g=nx.Graph()
    for atom in mol.iterateAtoms():
        g.add_node(atom.index(),symbol=normalize_text(atom.symbol()),charge=int(atom.charge() or 0),opaque=bool(atom.isPseudoatom()))
    for bond in mol.iterateBonds():
        g.add_edge(bond.source().index(),bond.destination().index(),label=int(bond.bondOrder())-1)
    return g


def normalize_bond_orders(ig,atoms,bonds):
    """Aromatic/Kekule normalization on each input independently, never GT snapping."""
    labels=[int(b['label']) for b in bonds];mol=ig.createMolecule();created=[]
    try:
        for atom in atoms:
            symbol=atom['symbol'] if atom['symbol'] in ATOM_CLASSES[1:] else '*'
            a=mol.addAtom(symbol);a.setCharge(int(atom.get('formal_charge',0)));created.append(a)
        native=[]
        for bond in bonds:
            label=int(bond['label']);order=1 if label in [4,5] else label+1
            native.append(created[bond['begin']].addBond(created[bond['end']],order))
        mol.aromatize()
        return [int(b.bondOrder())-1 if label not in [4,5] else label for b,label in zip(native,labels)],True
    except Exception:
        # Nonstandard source molecules are retained, not silently sanitized away.
        return labels,False


def property_graph(ig,atoms,bonds,texts=None):
    """Existing 13-class atoms + anonymous dummy nodes, charges and typed edges.

    Every class-0/unsupported label is anonymized, including rare elements that
    the model cannot distinguish from aliases. Text mode restores literal labels
    on those nodes. This is an explicit model-vocabulary projection.
    """
    g=nx.DiGraph();orders,aromatic_ok=normalize_bond_orders(ig,atoms,bonds)
    for i,a in enumerate(atoms):
        opaque=a['symbol'] not in ATOM_CLASSES[1:]
        token='*' if opaque else a['symbol']
        text=normalize_text(texts.get(i,'')) if opaque and texts is not None else ''
        g.add_node(i,token=token,charge=int(a.get('formal_charge',0)),text=text)
    arcs=defaultdict(list)
    for b,label in zip(bonds,orders):
        a,c=int(b['begin']),int(b['end'])
        if a==c:raise ValueError('Self bond')
        arcs[a,c].append(label)
        if label not in [4,5]:arcs[c,a].append(label)
    for (a,b),values in arcs.items():g.add_edge(a,b,kinds=tuple(sorted(values)))
    g.graph.update(aromatic_normalization_ok=aromatic_ok,bond_instances=len(bonds))
    return g


def exact_property_match(predicted,reference,with_text=False):
    if len(predicted)!=len(reference) or predicted.number_of_edges()!=reference.number_of_edges():return None
    # Counts reject most failures before invoking exact isomorphism search.
    def signature(g):return Counter((d['token'],d['charge'],d['text'] if with_text else '') for _,d in g.nodes(data=True))
    if signature(predicted)!=signature(reference):return None
    match=nx.algorithms.isomorphism.DiGraphMatcher(predicted,reference,
        node_match=lambda a,b:(a['token'],a['charge'],a['text'] if with_text else '')==
                              (b['token'],b['charge'],b['text'] if with_text else ''),
        edge_match=lambda a,b:a['kinds']==b['kinds'])
    return next(match.isomorphisms_iter(),None)


def alias_pattern(smiles):
    from rdkit import Chem
    mol=Chem.MolFromSmiles(smiles,sanitize=False)
    if mol is None:raise ValueError(smiles)
    g=nx.Graph()
    for a in mol.GetAtoms():g.add_node(a.GetIdx(),symbol=a.GetSymbol(),charge=a.GetFormalCharge())
    for b in mol.GetBonds():
        order=3 if b.GetIsAromatic() else {1.:0,2.:1,3.:2}[b.GetBondTypeAsDouble()]
        g.add_edge(b.GetBeginAtomIdx(),b.GetEndAtomIdx(),label=order)
    return g


def fold_expanded_reference(source,display,aliases,max_states=256):
    """Exact full-source coverage using predefined rooted alias expansions.

    Try literal labels first. For expanded CF3 etc, expand the *reference*
    display alias into a rooted dictionary pattern, require full graph
    isomorphism against source, then contract the proven source atom sets.
    Opaque R labels have no invented expansion. All atom/boundary coverage is
    checked. Partial MCS results are never accepted.
    """
    literal=simple_exact(display,source)
    if literal is not None:
        return dict(ok=True,expanded_aliases=[],source_atom_groups={str(i):[j] for i,j in literal.items()},
            source_atoms=len(source),display_atoms=len(display),states=1)
    candidates=[]
    for i,d in display.nodes(data=True):
        if d['symbol'] in aliases and d.get('opaque',d['symbol'] not in ATOM_CLASSES[1:]):
            try:
                pattern=alias_pattern(aliases[d['symbol']])
                if len(pattern)>1:candidates.append((i,pattern))
            except Exception:pass
    states=0
    choices=itertools.chain.from_iterable(itertools.combinations(range(len(candidates)),n) for n in range(len(candidates)+1))
    for choice in choices:
        if states>=max_states:break
        states+=1;expanded=display.copy();owner={i:i for i in expanded};next_id=max(expanded.nodes,default=-1)+1
        for ci in choice:
            node,pattern=candidates[ci];external=list(expanded.edges(node,data=True));attrs=expanded.nodes[node].copy()
            expanded.remove_node(node);owner.pop(node)
            remap={0:node}
            for j in pattern.nodes:
                if j:remap[j]=next_id;next_id+=1
            for j,props in pattern.nodes(data=True):
                values=dict(props)
                if j==0:values['charge']+=attrs['charge']
                expanded.add_node(remap[j],**values);owner[remap[j]]=node
            for u,v,props in pattern.edges(data=True):expanded.add_edge(remap[u],remap[v],**props)
            for u,v,props in external:expanded.add_edge(node,v if u==node else u,**props)
        if len(expanded)!=len(source):continue
        mapping=simple_exact(expanded,source)
        if mapping is None:continue
        groups=defaultdict(list)
        for i,j in mapping.items():groups[owner[i]].append(j)
        inverse={j:i for i,group in groups.items() for j in group}
        assert set(inverse)==set(source) and len(inverse)==len(source)
        collapsed=nx.Graph();collapsed.add_nodes_from(display.nodes(data=True))
        for u,v,props in source.edges(data=True):
            a,b=inverse[u],inverse[v]
            if a!=b:
                if collapsed.has_edge(a,b):raise ValueError('Multiple external bonds cannot collapse to one simple bond')
                collapsed.add_edge(a,b,**props)
        # Same display node identities: validate every retained boundary bond.
        assert {frozenset((u,v)):d['label'] for u,v,d in collapsed.edges(data=True)}=={frozenset((u,v)):d['label'] for u,v,d in display.edges(data=True)}
        return dict(ok=True,expanded_aliases=[candidates[i][0] for i in choice],
            source_atom_groups={str(i):sorted(v) for i,v in groups.items()},source_atoms=len(source),display_atoms=len(display),states=states)
    return dict(ok=False,reason='No exact full-coverage reference expansion match',states=states,source_atoms=len(source),display_atoms=len(display))


def mcs_diagnostic(atoms,bonds,reference_atoms,reference_bonds,timeout=1):
    """Partial chemical core diagnostic, ignoring charge and drawn wedge direction."""
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    def molecule(nodes,edges):
        m=Chem.RWMol()
        for a in nodes:
            atom=Chem.Atom(a['symbol'] if a['symbol'] in ATOM_CLASSES[1:] else '*')
            atom.SetFormalCharge(int(a.get('formal_charge',0)));m.AddAtom(atom)
        seen=set()
        for b in edges:
            u,v=b['begin'],b['end'];key=tuple(sorted((u,v)))
            if u==v or key in seen:continue
            seen.add(key);m.AddBond(u,v,{0:Chem.BondType.SINGLE,1:Chem.BondType.DOUBLE,2:Chem.BondType.TRIPLE,3:Chem.BondType.AROMATIC,4:Chem.BondType.SINGLE,5:Chem.BondType.SINGLE}[b['label']])
        mol=m.GetMol();Chem.GetSymmSSSR(mol);return mol
    try:
        a,b=molecule(atoms,bonds),molecule(reference_atoms,reference_bonds)
        result=rdFMCS.FindMCS([a,b],timeout=timeout,atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,ringMatchesRingOnly=True)
        query=Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
        ma=a.GetSubstructMatch(query) if query is not None else ();mb=b.GetSubstructMatch(query) if query is not None else ()
        return dict(atoms=result.numAtoms,bonds=result.numBonds,canceled=result.canceled,smarts=result.smartsString,
            predicted_to_reference=dict(zip(ma,mb)),used_for_exact_score=False,timeout_seconds=timeout)
    except Exception as exc:return dict(error=str(exc),used_for_exact_score=False)
