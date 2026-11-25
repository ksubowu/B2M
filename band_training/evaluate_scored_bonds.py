"""Reload v3 weights and independently evaluate line and band detections."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import Subset
from .train import load_model
from .continue_indigo import loader
from .train_scored_bonds import ScoredJointModel,SCHEMA,evaluate
from .scored_bond_model import ScoredBondDecoder
from .scored_bond_data import ScoredBondDataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full-val',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert s['schema']==SCHEMA;c=s['config'];device=torch.device('cuda:0')
    base=load_model(Path(c['base']),device).base;base.load_state_dict(s['base'])
    decoder=ScoredBondDecoder().to(device);decoder.load_state_dict(s['decoder']);model=ScoredJointModel(base,decoder).to(device)
    val=ScoredBondDataset(c['dataset'],'val')
    if a.full_val:ids=list(range(len(val)))
    else:
        selected=json.loads((a.checkpoint.parent/'validation_sample_ids.json').read_text());lookup={r['sample_id']:i for i,r in enumerate(val.records)}
        ids=[lookup[s] for s in selected]
    settings=SimpleNamespace(line_threshold=c['line_threshold'],band_threshold=c['band_threshold'])
    evaluate(model,loader(Subset(val,ids),c['batch_size'],c['workers']),device,settings,a.output,'reloaded')

if __name__=='__main__':main()
