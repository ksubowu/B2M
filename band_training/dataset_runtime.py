"""Resolve the separate chemistry package for optional molecular graph scoring."""
import importlib.util
import os
import sys
from pathlib import Path


def install_generator_path(dataset):
    explicit=os.environ.get('B2M_CHEM_PREP_PATH')
    if explicit:
        candidate=Path(explicit).expanduser().resolve()
        if not (candidate/'chem_prep/backend.py').is_file():raise FileNotFoundError(f'Invalid B2M_CHEM_PREP_PATH: {candidate}')
        sys.path.insert(0,str(candidate));return candidate
    spec=importlib.util.find_spec('chem_prep')
    if spec and spec.origin:return Path(spec.origin).resolve().parents[1]
    # Support datasets that already carry a pinned legacy renderer snapshot.
    root=Path(dataset)
    for candidate in [root/'generator_snapshot',root.parent/'chem_structure_prep']:
        if (candidate/'chem_prep/backend.py').is_file():
            sys.path.insert(0,str(candidate));return candidate
    raise ModuleNotFoundError('Molecular graph scoring requires the separate chem-structure-prep package. Install it from https://github.com/ksubowu/chem-structure-prep; neural prediction does not require it.')
