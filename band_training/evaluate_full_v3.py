"""Reload a paired checkpoint and evaluate on its saved smoke subset or full val."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import Subset
from .train import load_model
from .continue_indigo import JointModel,loader
from .train_full_v3 import evaluate
from .band_line_model import BandLineDecoder
from .full_v3_data import FullV3Dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full-val',action='store_true');p.add_argument('--line-association',default='exclusive_box_noisy')
    p.add_argument('--workers',type=int,default=4);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);device=torch.device('cuda:0')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if state['schema']!='paired_band_line_joint_v2':raise ValueError(state['schema'])
    config=state['config'];base=load_model(Path(config['base']),device).base;base.load_state_dict(state['base'])
    band=BandLineDecoder().to(device);band.load_state_dict(state['band']);model=JointModel(base,band).to(device)
    val=FullV3Dataset(config['dataset'],'val',endpoint_order=config.get('endpoint_order','canonical'))
    if args.full_val:ids=list(range(len(val)))
    else:
        selected=json.loads((args.checkpoint.parent/'validation_sample_ids.json').read_text());lookup={r['sample_id']:i for i,r in enumerate(val.records)}
        ids=[lookup[s] for s in selected]
    settings=SimpleNamespace(threshold=state['threshold'],endpoint_radius=config['endpoint_radius'],line_association=args.line_association)
    result=evaluate(model,loader(Subset(val,ids),config['batch_size'],args.workers),device,settings,args.output,'reloaded')
    if args.full_val:
        result['note']='All frozen validation images; EPE uses GT detection anchors, graph metrics use decoded predictions; no independent test claim.'
        (args.output/'reloaded_metrics.json').write_text(json.dumps(result,indent=2))
    (args.output/'scope.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint.resolve()),full_validation=args.full_val,
        images=len(ids),checkpoint_step=state['step'],line_association=args.line_association),indent=2))

if __name__=='__main__':main()
