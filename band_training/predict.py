"""Predict independent boxes/bands, then assemble a graph using geometry only."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torchvision.transforms import functional as TF
from .train import load_model,decode_atoms
from .model import BandDecoder,decode
from .geometry import suppress,link_many
from .charge_association import assign_charges

ATOM_CLASSES=['other','C','O','N','Cl','Br','S','F','B','I','P','H','Si']
BOND_CLASSES=['single','double','triple','aromatic','wedge','dash']
CHARGES={13:-4,14:-2,15:-1,16:1,17:2}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--association',choices=['geometry','kdtree','band_directional'],default='band_directional',
                   help='Directional band/box anchors by default; kdtree reproduces the original validation.')
    p.add_argument('--max-bands',type=int,default=0,
                   help='Maximum candidates before region NMS; 0 means no count limit (default).')
    p.add_argument('--charge-association',choices=['legacy','joint_valence'],default='joint_valence')
    args=p.parse_args();torch.set_num_threads(2)
    config=json.loads((args.run/'config.json').read_text());device=torch.device(args.device)
    base=load_model(Path(config['base']),device).base.eval()
    joint=(args.run/'best_joint.pt').is_file()
    if joint:
        checkpoint=torch.load(args.run/'best_joint.pt',map_location='cpu',weights_only=False)
        base.load_state_dict(checkpoint['base'],strict=True)
    elif (args.run/'atom_decoder.pth').exists():
        base.box_decoder.load_state_dict(torch.load(args.run/'atom_decoder.pth',map_location=device,weights_only=True))
    if not joint:checkpoint=torch.load(args.run/'best_band.pth',map_location=device,weights_only=False)
    decoder=BandDecoder().to(device).eval();decoder.load_state_dict(checkpoint['band'])
    original=Image.open(args.image).convert('RGB');w,h=original.size
    scale=min(640/w,640/h);nw,nh=round(w*scale),round(h*scale);ox,oy=(640-nw)//2,(640-nh)//2
    scale_xy=np.array([nw/w,nh/h])
    canvas=Image.new('RGB',(640,640),'white');canvas.paste(original.resize((nw,nh),Image.Resampling.LANCZOS),(ox,oy))
    tensor=TF.to_tensor(canvas)[None].to(device)
    with torch.no_grad():
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=joint and device.type=='cuda'):
            f=base.encoder(base.backbone(TF.normalize(tensor,[.485,.456,.406],[.229,.224,.225])))
        if isinstance(f,tuple):f=f[0]
        # Match the FP16 feature-cache precision used in pilot training/evaluation.
        f=[v.float() if joint else v.half().float() for v in f]
        boxes=decode_atoms(base.box_decoder(f))
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            band_output=decoder(f,tensor.mean(1,keepdim=True))
        band_candidates=decode(band_output,minimum=checkpoint['threshold'],topk=args.max_bands)
        bands=suppress(band_candidates)
    atoms=[a for a in boxes if a['label']<13];charges=[a for a in boxes if a['label']>=13]
    edges=[];unlinked=[]
    pairs=link_many([band['quad'] for band in bands],[a['bbox'] for a in atoms],args.association)
    for i,(band,pair) in enumerate(zip(bands,pairs)):
        if pair is None:unlinked.append(i)
        else:edges.append(dict(begin=pair[0],end=pair[1],type=BOND_CLASSES[band['label']],
                               confidence=band['score'],band_index=i))
    charge_links=assign_charges(atoms,charges,edges,method=args.charge_association)
    for charge,assignment in zip(charges,charge_links):
        charge['atom_index']=assignment['atom_index']
        charge['formal_charge']=CHARGES[charge['label']]
    # Convert all geometry back to the original image's coordinate system.
    for box in atoms+charges:
        x,y,bw,bh=box['bbox'];box['bbox']=[(x-ox)/scale_xy[0],(y-oy)/scale_xy[1],bw/scale_xy[0],bh/scale_xy[1]]
    for atom in atoms:atom['symbol']=ATOM_CLASSES[atom['label']]
    for band in bands:
        band['quad']=((band['quad']-np.array([ox,oy]))/scale_xy).tolist()
        band['type']=BOND_CLASSES[band['label']]
    graph=dict(image=str(args.image.resolve()),image_size=[w,h],atoms=atoms,charges=charges,bonds=edges,
               bands=bands,unlinked_band_indices=unlinked,threshold=checkpoint['threshold'],
               association_method=args.association,
               charge_association=args.charge_association,charge_assignment_audit=charge_links,
               max_band_candidates=args.max_bands,
               band_candidates_before_nms=len(band_candidates),
               association='Geometric postprocessing only; no atom-query pointer head.',
               warning='Experimental synthetic-data pilot; output is a graph, not validated SMILES.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(graph,indent=2))
    box_image=original.copy();band_image=original.copy()
    draw=ImageDraw.Draw(box_image)
    for box in atoms+charges:
        x,y,bw,bh=box['bbox'];draw.rectangle((x,y,x+bw,y+bh),outline='blue' if box['label']<13 else 'red',width=2)
    draw=ImageDraw.Draw(band_image)
    for band in bands:
        poly=[tuple(point) for point in band['quad']];draw.line(poly+[poly[0]],fill='orange',width=2)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    comparison=Image.new('RGB',(3*w,h+40),'white');draw=ImageDraw.Draw(comparison)
    for i,(im,title) in enumerate(zip([original,box_image,band_image],['Original','Predicted atom / charge boxes','Predicted bond regions'])):
        comparison.paste(im,(i*w,40));draw.text((i*w+8,8),title,font=font,fill='black')
    comparison.save(args.output.with_suffix('.png'))
    print(json.dumps(dict(atoms=len(atoms),charges=len(charges),bands=len(bands),linked_bonds=len(edges),
                          unlinked=len(unlinked),graph=str(args.output.with_suffix('.json'))),indent=2))

if __name__=='__main__':main()
