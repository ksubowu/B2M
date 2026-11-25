"""Image-only inference/export for independent line-score and band-score heads."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from .checkpoint_runtime import load_joint_base
from .train import load_model,decode_atoms
from .continue_indigo import write
from .train_scored_bonds import ScoredJointModel,SCHEMA,preview
from .scored_bond_model import ScoredBondDecoder,decode_scored_bonds
from .scored_bond_postprocess import assemble_scored_bonds,refine_peak_lines
from .mixed_data import letterbox_transform,ATOM_CLASSES
from .charge_association import assign_charges,CHARGES


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--line-threshold',type=float);p.add_argument('--band-threshold',type=float)
    p.add_argument('--line-association',choices=['forward','exclusive_box','exclusive_box_noisy'],default='forward')
    p.add_argument('--endpoint-cluster-radius',type=float,default=0.);p.add_argument('--line-nms-radius',type=float,default=3.)
    p.add_argument('--stereo-from-band',action='store_true');p.add_argument('--no-line-peaks',action='store_true')
    p.add_argument('--refine-line-peaks',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert s['schema']==SCHEMA;c=s['config'];device=torch.device(a.device)
    base=load_joint_base(s,device)
    decoder=ScoredBondDecoder().to(device);decoder.load_state_dict(s['decoder'],strict=True);model=ScoredJointModel(base,decoder).to(device).eval()
    with Image.open(a.image) as im:original=im.convert('RGB')
    scale,offset,shape=letterbox_transform(*original.size);canvas=Image.new('RGB',(640,640),'white')
    canvas.paste(original.resize(shape,Image.Resampling.LANCZOS),tuple(offset));image=TF.to_tensor(canvas).to(device)
    with torch.no_grad():boxes,output=model(image[None])
    lt=c['line_threshold'] if a.line_threshold is None else a.line_threshold;bt=c['band_threshold'] if a.band_threshold is None else a.band_threshold
    if a.no_line_peaks and a.refine_line_peaks:p.error('--refine-line-peaks retains one candidate per peak; incompatible with --no-line-peaks')
    use_peaks=not (a.no_line_peaks or a.refine_line_peaks)
    decoded=decode_scored_bonds(output,line_threshold=lt,band_threshold=bt,apply_line_peaks=use_peaks)
    high_band=decode_scored_bonds(output,line_threshold=lt,band_threshold=.999,apply_line_peaks=use_peaks)
    high_line=decode_scored_bonds(output,line_threshold=.999,band_threshold=bt,apply_line_peaks=use_peaks)
    assert [x['query_id'] for x in decoded['lines']]==[x['query_id'] for x in high_band['lines']]
    assert [x['query_id'] for x in decoded['bands']]==[x['query_id'] for x in high_line['bands']]
    ap=decode_atoms(boxes);atoms=[x for x in ap if x['label']<13];charges=[x for x in ap if x['label']>=13]
    if a.refine_line_peaks:decoded=refine_peak_lines([x['bbox'] for x in atoms],decoded,lt)
    graph=assemble_scored_bonds([x['bbox'] for x in atoms],decoded,line_nms_radius=a.line_nms_radius,
        line_association=a.line_association,endpoint_cluster_radius=a.endpoint_cluster_radius,stereo_from_band=a.stereo_from_band)
    assignments=assign_charges(atoms,charges,graph['bonds'],method='safe_adaptive',lines=[x['line'] for x in decoded['lines']])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    preview(image,dict(record=dict(sample_id=a.image.name)),atoms,charges,decoded,a.output.with_suffix('.png'))
    for atom in atoms:atom.update(symbol=ATOM_CLASSES[atom['label']],formal_charge=0)
    for charge,assignment in zip(charges,assignments):
        charge.update(value=CHARGES[charge['label']],atom_index=assignment['atom_index'])
        if assignment['atom_index'] is not None:atoms[assignment['atom_index']]['formal_charge']+=charge['value']
    def points(values):return ((np.asarray(values)-offset)/scale).tolist()
    for box in atoms+charges:
        x,y,w,h=box['bbox'];box['bbox']=[*((np.array([x,y])-offset)/scale).tolist(),*(np.array([w,h])/scale).tolist()]
    lines=[dict(x,line=points(x['line'])) for x in decoded['lines']]
    bands=[dict(x,quad=points(x['quad'])) for x in decoded['bands']]
    probes=[dict(x,quad=points(x['quad'])) for x in decoded['paired_band_probes']]
    typed=[dict(x,line=points(x['line']),quad=points(x['quad'])) for x in graph['instances']]
    result=dict(schema='independent_scored_line_band_graph_v3',image=str(a.image.resolve()),image_size=list(original.size),
        coordinate_frame='original_image_pixels',atoms=atoms,charges=charges,lines=lines,bands=bands,paired_band_probes=probes,
        line_graph=[dict(x,line=points(x['line'])) for x in graph['line_graph']],bonds=graph['bonds'],typed_instances=typed,
        type_pending_lines=[dict(x,line=points(x['line'])) for x in graph['untyped_lines']],
        unlinked_lines=[dict(x,line=points(x['line'])) for x in graph['unlinked_lines']],
        unlinked_typed_indices=graph['unlinked_band_indices'],band_only_query_ids=[x['query_id'] for x in graph['band_only_candidates']],
        line_threshold=lt,band_threshold=bt,score_definitions=c['score_definitions'],pairing=c['pairing'],
        postprocess=dict(line_association=a.line_association,endpoint_cluster_radius=a.endpoint_cluster_radius,line_nms_radius=a.line_nms_radius,
                        stereo_from_band=a.stereo_from_band,line_local_peaks=not a.no_line_peaks,refine_line_peaks=a.refine_line_peaks,stereo_flips=len(graph['stereo_band_direction_flips'])),
        independence_check=dict(line_detections_unchanged_by_band_threshold=True,band_detections_unchanged_by_line_threshold=True),
        uses_ground_truth=False,uses_third_party_ocr=False,max_candidate_count=None)
    write(a.output.with_suffix('.json'),result)
    print(dict(lines=len(lines),bands=len(bands),typed_bonds=len(graph['bonds']),type_pending=len(graph['untyped_lines']),path=str(a.output.with_suffix('.json'))),flush=True)

if __name__=='__main__':main()
