"""Paired targets with independent line/band scores at shared instance queries."""
from .full_v3_data import FullV3Dataset
from .scored_bond_model import scored_targets


class ScoredBondDataset(FullV3Dataset):
    def __getitem__(self,index):
        image,meta=super().__getitem__(index)
        meta['scored_bonds']=scored_targets(meta['labels'],meta['quads'],meta['lines'])
        return image,meta
