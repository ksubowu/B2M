"""Locate the immutable renderer snapshot in legacy and v6 dataset layouts."""
import sys
from pathlib import Path


def install_generator_path(dataset):
    root=Path(dataset)
    for candidate in [root/'generator_snapshot',root.parent/'chem_structure_prep',Path(__file__).resolve().parents[1]/'chem_structure_prep']:
        if (candidate/'chem_prep/backend.py').is_file():
            sys.path.insert(0,str(candidate));return candidate
    raise FileNotFoundError(f'No local chem_prep renderer snapshot for {root}')
