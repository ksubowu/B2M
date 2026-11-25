"""Reevaluate unchanged model detections/edges with revised charge association."""
import argparse,json
from pathlib import Path
import torch
from .continue_indigo import load_model,JointModel,loader,evaluate,write,sha
from .mixed_data import MixedIndigoDataset
from .model import BandDecoder

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    cfg=json.loads((args.run/'config.json').read_text());state=torch.load(args.run/'best_joint.pt',map_location='cpu',weights_only=False)
    device=torch.device('cuda:0');base=load_model(Path(cfg['base']),device).base;base.load_state_dict(state['base'],strict=True)
    band=BandDecoder().to(device);band.load_state_dict(state['band'],strict=True);model=JointModel(base,band).to(device).eval()
    dataset=Path(cfg['dataset'])
    if not dataset.exists():dataset=Path(cfg['source_dataset'])
    ds=MixedIndigoDataset(dataset,'val');old=json.loads(args.reference.read_text())
    new=evaluate(model,loader(ds,8,4),device,state['threshold'],args.output,'joint_charge',association='band_directional',charge_association='joint_valence')
    assert old['atom_f1']==new['atom_f1'] and old['charge_f1']==new['charge_f1']
    assert old['bond_graph']==new['bond_graph'],'Only charge ownership may change in this regression'
    keys=['atom_f1','charge_f1','charge_graph_exact','charged_image_exact','full_graph_exact']
    comparison=dict(checkpoint=str(args.run/'best_joint.pt'),checkpoint_sha256=sha(args.run/'best_joint.pt'),
        old={k:old[k] for k in keys},new={k:new[k] for k in keys},unchanged_detections_and_bonds=True)
    write(args.output/'comparison.json',comparison);print(json.dumps(comparison,indent=2),flush=True)

if __name__=='__main__':main()
