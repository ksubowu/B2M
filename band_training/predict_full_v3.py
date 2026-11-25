"""Image-only inference for paired-band-line-v2 checkpoints and graph assembly."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from .train import load_model,decode_atoms
from .continue_indigo import JointModel,write
from .band_line_model import BandLineDecoder
from .model import decode
from .mixed_data import letterbox_transform,ATOM_CLASSES
from .geometry import suppress,assemble_band_lines
from .charge_association import assign_charges,CHARGES
from .train_full_v3 import preview


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--endpoint-radius',type=float,default=None);args=p.parse_args();torch.set_num_threads(2)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if state.get('schema')!='paired_band_line_joint_v2':raise ValueError('Requires paired_band_line_joint_v2 weights')
    config=state['config'];device=torch.device(args.device)
    base=load_model(Path(config['base']),device).base;base.load_state_dict(state['base'],strict=True)
    band=BandLineDecoder().to(device);band.load_state_dict(state['band'],strict=True)
    model=JointModel(base,band).to(device).eval()
    if device.type!='cuda':raise ValueError('Current shared JointModel uses CUDA autocast; use a CUDA device')
    with Image.open(args.image) as handle:original=handle.convert('RGB')
    scale,offset,shape=letterbox_transform(*original.size)
    canvas=Image.new('RGB',(640,640),'white');canvas.paste(original.resize(shape,Image.Resampling.LANCZOS),tuple(offset))
    image=TF.to_tensor(canvas).to(device)
    with torch.no_grad():boxes,out=model(image[None])
    ap=decode_atoms(boxes);atoms=[a for a in ap if a['label']<13];charges=[a for a in ap if a['label']>=13]
    instances=suppress(decode(out,minimum=state['threshold']))
    radius=config.get('endpoint_radius',2.) if args.endpoint_radius is None else args.endpoint_radius
    graph=assemble_band_lines([a['bbox'] for a in atoms],instances,line_association=config.get('line_association','exclusive_box_noisy'),endpoint_cluster_radius=radius)
    assigned=assign_charges(atoms,charges,graph['bonds'],method='safe_adaptive',lines=[b['line'] for b in instances])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    preview(image,dict(record=dict(sample_id=args.image.name)),atoms,charges,instances,args.output.with_suffix('.png'))
    for atom in atoms:atom.update(symbol=ATOM_CLASSES[atom['label']],formal_charge=0)
    for charge,a in zip(charges,assigned):
        charge.update(value=CHARGES[charge['label']],atom_index=a['atom_index'])
        if a['atom_index'] is not None:atoms[a['atom_index']]['formal_charge']+=charge['value']
    # All emitted geometric fields are returned to the original image frame.
    for box in atoms+charges:
        b=np.asarray(box['bbox'],float);box['bbox']=[*((b[:2]-offset)/scale).tolist(),*(b[2:]/scale).tolist()]
    for b in instances:
        for key in ['quad','line']:b[key]=((np.asarray(b[key])-offset)/scale).tolist()
    audit=graph['endpoint_assignment_audit']
    if audit is not None:
        audit['unique_endpoint_coordinates']=((np.asarray(audit['unique_endpoint_coordinates']).reshape(-1,2)-offset)/scale).tolist()
        audit['cluster_radius_letterbox_px']=audit.pop('cluster_radius')
    result=dict(schema='paired_band_line_graph_v2',image=str(args.image.resolve()),image_size=list(original.size),
        coordinate_frame='original_image_pixels',atoms=atoms,charges=charges,bond_instances=instances,bonds=graph['bonds'],
        unlinked_band_indices=graph['unlinked_band_indices'],endpoint_assignment_audit=audit,charge_assignment_audit=assigned,
        charge_audit_score_coordinates='640 letterbox pixels',threshold=state['threshold'],max_band_candidates=0,
        uses_ground_truth=False,uses_third_party_ocr=False,
        note='Experimental graph in existing 13 atom classes; other labels are not transcribed or expanded into SMILES.')
    write(args.output.with_suffix('.json'),result)
    print(json.dumps(dict(atoms=len(atoms),charges=len(charges),bands=len(instances),bonds=len(graph['bonds']),
        graph=str(args.output.with_suffix('.json')),comparison=str(args.output.with_suffix('.png')))),flush=True)

if __name__=='__main__':main()
