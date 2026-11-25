"""Literal-label graph equality, invariant to symmetric SMILES traversal."""
from rdkit import Chem


def label_graph_equal(predicted, truth):
    """Canonical CX strings can differ across symmetric atom traversals.

    Assign deterministic artificial isotope colors to literal labels on copies
    before canonical graph comparison. This retains label attachment and stereo,
    and does not count chemically different raw labels as identical.
    This benchmark contains no real isotopes.
    """
    if predicted is None or truth is None:
        return False
    words = sorted({a.GetProp('atomLabel') for m in [predicted, truth]
                    for a in m.GetAtoms() if a.HasProp('atomLabel')})
    signatures = []
    for mol in [predicted, truth]:
        colored = Chem.Mol(mol)
        for atom in colored.GetAtoms():
            if atom.GetIsotope():
                raise ValueError('Label-color scoring only supports isotope-free truth')
            if atom.HasProp('atomLabel'):
                atom.SetIsotope(100+words.index(atom.GetProp('atomLabel')))
        signatures.append(Chem.MolToSmiles(colored, isomericSmiles=True))
    return signatures[0] == signatures[1]

def specified_stereo_compatible(predicted, truth):
    if predicted is None or truth is None:
        return False
    if Chem.MolToSmiles(predicted,isomericSmiles=False)!=Chem.MolToSmiles(truth,isomericSmiles=False):
        return False
    return predicted.HasSubstructMatch(truth,useChirality=True)
