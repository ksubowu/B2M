import sys,unittest
from pathlib import Path
import networkx as nx
from .molecular_graph_metrics import (property_graph,exact_property_match,native_graph,fold_expanded_reference,normalize_text)

from chem_prep.backend import Indigo


class MolecularGraphMetricsTests(unittest.TestCase):
    def setUp(self):self.ig=Indigo()

    def test_anonymous_alias_then_ocr_label_comparison(self):
        atoms=[dict(symbol='C'),dict(symbol='CF3')];bonds=[dict(begin=0,end=1,label=0)]
        a=property_graph(self.ig,atoms,bonds,texts={1:'CF₃'})
        b=property_graph(self.ig,list(reversed(atoms)),[dict(begin=1,end=0,label=0)],texts={0:'R1'})
        self.assertIsNotNone(exact_property_match(a,b))
        self.assertIsNone(exact_property_match(a,b,with_text=True))
        c=property_graph(self.ig,atoms,bonds,texts={1:'CF3'})
        self.assertIsNotNone(exact_property_match(a,c,with_text=True))

    def test_charge_and_directed_stereo_are_required(self):
        atoms=[dict(symbol='C'),dict(symbol='N')]
        a=property_graph(self.ig,atoms,[dict(begin=0,end=1,label=4)])
        b=property_graph(self.ig,atoms,[dict(begin=1,end=0,label=4)])
        self.assertIsNone(exact_property_match(a,b))
        charged=[dict(symbol='C'),dict(symbol='N',formal_charge=1)]
        c=property_graph(self.ig,charged,[dict(begin=0,end=1,label=4)])
        self.assertIsNone(exact_property_match(a,c))

    def test_aromatic_and_kekule_are_normalized(self):
        atoms=[dict(symbol='C') for _ in range(6)]
        a=property_graph(self.ig,atoms,[dict(begin=i,end=(i+1)%6,label=i%2) for i in range(6)])
        b=property_graph(self.ig,atoms,[dict(begin=i,end=(i+1)%6,label=3) for i in range(6)])
        self.assertIsNotNone(exact_property_match(a,b))

    def test_fold_cf3_requires_all_fluorines_and_boundary(self):
        display=nx.Graph();display.add_node(0,symbol='C',charge=0,opaque=False);display.add_node(1,symbol='CF3',charge=0,opaque=True)
        display.add_edge(0,1,label=0)
        source=native_graph(self.ig.loadMolecule('CC(F)(F)F'))
        result=fold_expanded_reference(source,display,{'CF3':'[C](F)(F)F'})
        self.assertTrue(result['ok']);self.assertEqual(len(result['source_atom_groups']['1']),4)
        self.assertEqual(sorted(sum(result['source_atom_groups'].values(),[])),list(range(5)))
        for wrong in ['CC(F)F','CC(F)(Cl)F','CCC(F)(F)F']:
            self.assertFalse(fold_expanded_reference(native_graph(self.ig.loadMolecule(wrong)),display,{'CF3':'[C](F)(F)F'})['ok'])

    def test_real_unsupported_element_is_not_expanded_as_alias(self):
        source=native_graph(self.ig.loadMolecule('CCC'))
        display=nx.Graph();display.add_node(0,symbol='Pr',charge=0,opaque=False)
        self.assertFalse(fold_expanded_reference(source,display,{'Pr':'CCC'})['ok'])

    def test_text_normalization_does_not_conflate_r_suffixes(self):
        self.assertEqual(normalize_text(' R₁ '),'R1')
        self.assertNotEqual(normalize_text('R1'),normalize_text('R2'))
        self.assertNotEqual(normalize_text('Cl'),normalize_text('CI'))

    def test_missing_or_duplicate_bond_never_matches(self):
        atoms=[dict(symbol='C'),dict(symbol='O')];bonds=[dict(begin=0,end=1,label=0)]
        reference=property_graph(self.ig,atoms,bonds)
        for edges in [[],bonds+bonds]:
            self.assertIsNone(exact_property_match(property_graph(self.ig,atoms,edges),reference))

    def test_native_reference_formats_and_rsite_taxonomy(self):
        from rdkit import Chem
        from .audit_graph_references import reference_from_annotation
        # Generated in-memory references exercise both molfile formats without
        # depending on a private validation dataset or renderer snapshot.
        molecule=Chem.MolFromSmiles('C[W].[Y].[Ar].*')
        symbols=['C','W','Y','Ar','R11']
        for v3000 in [False,True]:
            ann=dict(rendered_molfile=Chem.MolToMolBlock(molecule,forceV3000=v3000),
                atoms=[dict(id=i,symbol=symbol,formal_charge=0,is_pseudoatom=i==4) for i,symbol in enumerate(symbols)],
                bonds=[dict(source=b.GetBeginAtomIdx(),target=b.GetEndAtomIdx()) for b in molecule.GetBonds()])
            _,atoms,bonds=reference_from_annotation(self.ig,ann)
            self.assertEqual(len(atoms),len(ann['atoms']));self.assertEqual(len(bonds),len(ann['bonds']))
            for atom in atoms:
                if atom['symbol']=='R11':self.assertTrue(atom['is_group'])
                if atom['symbol'] in ['W','Y','Ar']:self.assertFalse(atom['is_group'])

    def test_functional_metric_preserves_true_element_identity(self):
        from .rescore_functional_groups import functional_reference
        pred=property_graph(self.ig,[dict(symbol='*')],[])
        reference=property_graph(self.ig,[dict(symbol='Sb')],[],texts={0:'Sb'})
        self.assertIsNotNone(exact_property_match(pred,reference))
        strict=functional_reference(reference,[dict(atom_id=0,label='Sb',kind='unsupported_element')])
        self.assertIsNone(exact_property_match(pred,strict))
        alias=functional_reference(reference,[dict(atom_id=0,label='R1',kind='alias_or_rsite')])
        self.assertIsNotNone(exact_property_match(pred,alias))

if __name__=='__main__':unittest.main()
