"""Shared predicted-graph to RDKit assembly for CLI and inference notebook."""
from rdkit import Chem


def build_rdkit_molecule(graph, infer_double_stereo=False):
    rw = Chem.RWMol()
    index_map = {}
    warnings = ['OCR缩写另行处理；同位素尚未识别']
    if not infer_double_stereo:
        warnings.append('未启用二维几何双键立体信息恢复')
    for a in graph['atoms']:
        atom_obj = Chem.Atom(a['symbol'])
        atom_obj.SetFormalCharge(int(a['formal_charge']))
        index_map[a['index']] = rw.AddAtom(atom_obj)
        if a['symbol'] == '*': warnings.append(f"未知原子: {a['index']}")
    types = {0:Chem.BondType.UNSPECIFIED, 1:Chem.BondType.SINGLE,
             2:Chem.BondType.DOUBLE, 3:Chem.BondType.TRIPLE,
             4:Chem.BondType.AROMATIC, 5:Chem.BondType.SINGLE, 6:Chem.BondType.SINGLE}
    for b in graph['bonds']:
        i,j = index_map[b['begin_atom_idx']], index_map[b['end_atom_idx']]
        rw.AddBond(i,j,types[b['label_id']])
        bond_obj = rw.GetBondBetweenAtoms(i,j)
        if b['label_id'] == 5: bond_obj.SetBondDir(Chem.BondDir.BEGINWEDGE)
        if b['label_id'] == 6: bond_obj.SetBondDir(Chem.BondDir.BEGINDASH)
        if b['label_id'] == 4:
            rw.GetAtomWithIdx(i).SetIsAromatic(True)
            rw.GetAtomWithIdx(j).SetIsAromatic(True)
        if b['label_id'] == 0: warnings.append(f"未知键型: {b['index']}")
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms()); conf.Set3D(False)
    for a in graph['atoms']:
        x,y = a['center']
        conf.SetAtomPosition(index_map[a['index']], (float(x)/40, -float(y)/40, 0.))
    mol.AddConformer(conf)
    if mol.GetNumAtoms() == 0:
        return mol, dict(valid=False, smiles=None, error='没有预测原子', warnings=warnings)
    try:
        Chem.SanitizeMol(mol)
        Chem.AssignChiralTypesFromBondDirs(mol, confId=0, replaceExistingTags=True)
        if infer_double_stereo:
            # RDKit's 2D bond-perception path preserves wedge-derived atom tags.
            # Geometry cannot reveal whether an input SMILES intentionally left
            # stereo unspecified. Keep this opt-in and report its provenance.
            Chem.DetectBondStereochemistry(mol, 0)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return mol, dict(valid=True, smiles=smiles, error=None, warnings=warnings,
                         double_stereo_source='predicted_2d_geometry' if infer_double_stereo else 'disabled')
    except Exception as exc:
        return mol, dict(valid=False, smiles=None, error=f'{type(exc).__name__}: {exc}', warnings=warnings)
