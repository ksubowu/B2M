"""Audit display/source references independently of any predicted graph or OCR."""
import argparse,csv,json,sys,time,re
from pathlib import Path
from collections import Counter
from rdkit import Chem
from .molecular_graph_metrics import native_graph,simple_exact,fold_expanded_reference,property_graph,exact_property_match
from .continue_indigo import write

PERIODIC_SYMBOLS=frozenset(Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1,119))


def reference_from_annotation(ig,ann):
    lines=ann['rendered_molfile'].splitlines()
    bonds=[]
    # V2000 serializes pseudoatom labels A/X as query-reserved symbols. Load
    # neutral placeholders, then restore the recorded literal reference labels.
    # Connectivity/charges/stereo still come from the serialized molfile.
    if 'V2000' in lines[3]:
        n=int(lines[3][:3]);m=int(lines[3][3:6]);assert n==len(ann['atoms'])
        for i,a in enumerate(ann['atoms']):
            if a['is_pseudoatom']:lines[4+i]=lines[4+i][:31]+'C  '+lines[4+i][34:]
        for line in lines[4+n:4+n+m]:
            source=int(line[:3])-1;target=int(line[3:6])-1;order=int(line[6:9]);stereo=int(line[9:12].strip() or 0)
            label=4 if stereo==1 else 5 if stereo==6 else order-1
            bonds.append(dict(begin=source,end=target,label=label))
    elif 'V3000' in lines[3]:
        state=None
        for j,line in enumerate(lines):
            if line.endswith('BEGIN ATOM'):state='atom';continue
            if line.endswith('BEGIN BOND'):state='bond';continue
            if line.endswith('END ATOM') or line.endswith('END BOND'):state=None;continue
            if state=='atom':
                match=re.match(r'(M\s+V30\s+)(\d+)(\s+)("[^"]*"|\S+)',line)
                if match is None:raise ValueError('Unsupported continued V3000 atom record')
                i=int(match.group(2))-1
                if ann['atoms'][i]['is_pseudoatom']:lines[j]=line[:match.start(4)]+'C'+line[match.end(4):]
            elif state=='bond':
                fields=line.split()[2:];order=int(fields[1]);source=int(fields[2])-1;target=int(fields[3])-1
                cfg=next((int(x.split('=')[1]) for x in fields[4:] if x.startswith('CFG=')),0)
                label=4 if cfg==1 else 5 if cfg==3 else order-1
                bonds.append(dict(begin=source,end=target,label=label))
    else:raise ValueError('Unknown reference molfile version')
    mol=ig.loadMolecule('\n'.join(lines)+'\n')
    for i,a in enumerate(ann['atoms']):
        if a['is_pseudoatom']:
            atom=mol.getAtom(i);charge=int(atom.charge() or 0);atom.resetAtom(a['symbol']);atom.setCharge(charge)
    native=list(mol.iterateAtoms())
    assert len(native)==len(ann['atoms'])
    atoms=[]
    for i,(a,source) in enumerate(zip(native,ann['atoms'])):
        assert a.index()==i and source['id']==i
        assert a.symbol()==source['symbol']
        assert int(a.charge() or 0)==int(source['formal_charge'])
        atoms.append(dict(symbol=a.symbol(),formal_charge=int(a.charge() or 0),is_pseudoatom=bool(a.isPseudoatom()),
            is_group=bool(a.isPseudoatom() or a.symbol() not in PERIODIC_SYMBOLS)))
    # Raw bond records preserve wedges that chemical parsers may discard on
    # non-stereogenic/symmetric atoms. They remain part of this drawn-graph score.
    return mol,atoms,bonds


def heavy_graph(graph):
    g=graph.copy()
    # Provenance check only: original/augmented source may serialize implicit H
    # differently. This projection is NOT used for the scored output graphs.
    g.remove_nodes_from([i for i,d in g.nodes(data=True) if d['symbol']=='H' and not d.get('opaque',False) and g.degree(i)<=1])
    return g


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from .dataset_runtime import install_generator_path
    install_generator_path(a.dataset)
    from chem_prep.backend import Indigo
    from chem_prep.chemistry import load_molecule
    from repo.engine.rtv4.chem_ocr_expand import ABBREVIATIONS
    ig=Indigo();ig.setOption('ignore-stereochemistry-errors',True)
    stats=Counter();rows=[];errors=[];started=time.time()
    if a.manifest:records=json.loads(a.manifest.read_text())
    else:
        with (a.dataset/'val.csv').open() as f:records=list(csv.DictReader(f))
    for row in records:
        ann=json.loads((a.dataset/row['annotation']).read_text());detail=dict(sample_id=row['sample_id'])
        try:
            mol,atoms,bonds=reference_from_annotation(ig,ann)
            rendered_smiles=ig.loadMolecule(ann['SMILES'])
            serialized_ok=simple_exact(native_graph(mol),native_graph(rendered_smiles)) is not None
            assert serialized_ok,'Rendered SMILES and molfile disagree'
            byid={b['bond_id']:b['category_id']-1 for b in ann['bands']}
            annotation_bonds=[dict(begin=b['source'],end=b['target'],label=byid[b['id']]) for b in ann['bonds']]
            g=property_graph(ig,atoms,bonds);oracle=property_graph(ig,ann['atoms'],annotation_bonds)
            assert exact_property_match(oracle,g) is not None,'GT annotation does not reproduce reference graph'
            expected=Counter((q['atom_id'],q['value']) for q in ann['charges'])
            actual=Counter((i,x['formal_charge']) for i,x in enumerate(atoms) if x['formal_charge'])
            assert expected==actual,'Charge boxes and formal charges disagree'
            parent,_=load_molecule(ig,ann['source_SMILES']);source_graph=heavy_graph(native_graph(parent))
            display_graph=native_graph(mol)
            aug=ann.get('augmentation') or {}
            if aug.get('applied'):
                display_graph.remove_node(aug['dummy_atom_id']);stats['augmented_parent_restored']+=1
            folded=fold_expanded_reference(source_graph,heavy_graph(display_graph),ABBREVIATIONS)
            detail.update(rendered_smiles_verified=True,gt_star_exact=True,parent_source_fold=folded,
                dummy_labels=[dict(atom_id=i,label=x['symbol'],kind='alias_or_rsite' if x['is_group'] else 'unsupported_element') for i,x in enumerate(atoms) if g.nodes[i]['token']=='*'])
            stats.update(reference_valid=1,gt_star_exact=1,parent_source_verified=int(folded['ok']),expanded_source_aliases_folded=len(folded.get('expanded_aliases',[])),
                dummy_images=int(bool(detail['dummy_labels'])),dummy_nodes=len(detail['dummy_labels']),
                alias_or_rsite_nodes=sum(x['kind']=='alias_or_rsite' for x in detail['dummy_labels']),unsupported_element_nodes=sum(x['kind']=='unsupported_element' for x in detail['dummy_labels']))
        except Exception as exc:
            detail['error']=str(exc);errors.append(detail)
        rows.append(detail)
        if len(rows)%128==0:print('REFERENCE_AUDIT',len(rows),'/',len(records),flush=True)
    report=dict(status='passed' if not errors and stats['parent_source_verified']==len(rows) else 'needs_review',images=len(rows),counts=dict(stats),errors=errors,seconds=time.time()-started,
        scope='Ground-truth/reference-only audit. Rendered SMILES is the actual augmented image reference; parent restoration is provenance-only. Source check covers labels/charges/connectivity/aromatic orders, not full CIP/isotope equivalence.')
    write(a.output/'report.json',report);write(a.output/'samples.json',rows);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
