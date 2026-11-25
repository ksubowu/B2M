"""Freeze final image-only predictions and independent component scores."""
import argparse,gzip,json,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from .checkpoint_runtime import load_joint_base
from .train import load_model,decode_atoms,charge_exact
from .continue_indigo import JointModel,loader,match_boxes,write
from .full_v3_data import FullV3Dataset
from .band_line_model import BandLineDecoder
from .model import decode
from .geometry import suppress
from .train_full_v3 import metric_row
from .charge_association import assign_charges


def encode(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['train','val'],default='val');p.add_argument('--unique-assets',action='store_true')
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    device=torch.device('cuda:0');s=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=s['config']
    base=load_joint_base(s,device)
    scored=s.get('schema')=='independent_scored_line_band_joint_v3'
    if scored:
        from .scored_bond_model import ScoredBondDecoder,decode_scored_bonds
        from .scored_bond_postprocess import assemble_scored_bonds,refine_peak_lines
        from .train_scored_bonds import ScoredJointModel,typed_metric
        band=ScoredBondDecoder().to(device);band.load_state_dict(s['decoder']);model=ScoredJointModel(base,band).eval().to(device)
    else:
        band=BandLineDecoder().to(device);band.load_state_dict(s['band']);model=JointModel(base,band).eval().to(device)
    manifest=c.get(a.split+'_manifest')
    if manifest:
        manifest=Path(manifest)
        if not manifest.is_absolute():manifest=Path(c['dataset'])/manifest
    val=FullV3Dataset(c['dataset'],a.split,endpoint_order=c.get('endpoint_order','canonical'),manifest=manifest)
    input_rows=len(val);seen={}
    if a.unique_assets:
        for record in val.records:
            sid=record.get('asset_sample_id') or record['sample_id'].split('::repeat_')[0]
            if sid in seen:
                assert all(seen[sid][k]==record[k] for k in ['image','annotation','band_target','line_target']),'Repeated ID has different assets'
            else:seen[sid]=dict(record,sample_id=sid)
        val.records=list(seen.values())
    write(a.output/'evaluated_manifest.json',val.records)
    counts=Counter();started=time.time();tmp=a.output/'predictions.jsonl.gz.tmp';rows=[]
    method='independent_scored_heads' if scored else c['line_association']
    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=3) as out,torch.no_grad():
        for images,metas in loader(val,c['batch_size'],a.workers):
            boxes,bands=model(images.to(device))
            for bi,meta in enumerate(metas):
                ap=decode_atoms({k:v[bi:bi+1] for k,v in boxes.items() if torch.is_tensor(v) and v.ndim})
                atoms=[v for v in ap if v['label']<13];charges=[v for v in ap if v['label']>=13]
                ann=meta['ann'];mapping=match_boxes(atoms,ann['atoms']);cm=match_boxes(charges,ann['charges'],min_iou=.25)
                if scored:
                    refine=c.get('refine_line_peaks',False)
                    decoded=decode_scored_bonds(bands,bi,c['line_threshold'],c['band_threshold'],apply_line_peaks=not refine)
                    if refine:decoded=refine_peak_lines([a['bbox'] for a in atoms],decoded,c['line_threshold'])
                    graph=assemble_scored_bonds([a['bbox'] for a in atoms],decoded,stereo_from_band=c.get('stereo_from_band',False));instances=graph['instances']
                    bond=typed_metric(graph,ann,mapping);charge_lines=[b['line'] for b in decoded['lines']]
                else:
                    decoded=None;instances=suppress(decode(bands,bi,minimum=s['threshold']))
                    bond,graph=metric_row(instances,ann,atoms,mapping,method,c['endpoint_radius']);charge_lines=[b['line'] for b in instances]
                assignments=assign_charges(atoms,charges,graph['bonds'],method='safe_adaptive',lines=charge_lines)
                atom_ok=len(mapping)==len(atoms)==len(ann['atoms'])
                charge_ok=bool(charge_exact(ap,atoms,mapping,ann,assignments))
                charge_detection_ok=len(cm)==len(charges)==len(ann['charges'])
                score=dict(atom_exact=atom_ok,charge_detection_exact=charge_detection_ok,charge_owner_exact=charge_ok,
                    bond_exact=bool(bond['exact']),legacy_full_exact=bool(atom_ok and charge_ok and bond['exact']))
                counts.update(images=1,charged_images=int(bool(ann['charges'])),atom_tp=len(mapping),atom_pred=len(atoms),atom_gt=len(ann['atoms']),
                    charge_tp=len(cm),charge_pred=len(charges),charge_gt=len(ann['charges']),bond_tp=bond['tp'],bond_fp=bond['fp'],bond_fn=bond['fn'])
                counts.update({k:int(v) for k,v in score.items()})
                if ann['charges']:
                    counts.update(charged_detection_exact=int(charge_detection_ok),charged_owner_exact=int(charge_ok))
                else:counts.update(neutral_images=1,neutral_charge_exact=int(charge_ok))
                row=dict(sample_id=meta['record']['sample_id'],record=meta['record'],ann=ann,atoms=atoms,charges=charges,
                    instances=instances,bonds=graph['bonds'],unlinked=graph['unlinked_band_indices'],charge_assignments=assignments,
                    atom_mapping=mapping,charge_detection_mapping=cm,components=score,bond_metric=bond,
                    untyped_lines=graph.get('untyped_lines',[]),independent_heads=decoded)
                out.write(json.dumps(row,ensure_ascii=False,default=encode)+'\n');rows.append(dict(sample_id=row['sample_id'],**score))
            if counts['images']%128<len(metas):
                print('PREDICTED',counts['images'],'/',len(val),flush=True)
                write(a.output/'progress.json',dict(status='predicting',images=counts['images'],total=len(val)))
    tmp.replace(a.output/'predictions.jsonl.gz')
    saved_metrics=s.get('metrics',{})
    expected=saved_metrics.get('counts',{}).get('full_graph_exact') if scored else saved_metrics.get('counts',{}).get('forward_full_exact') if method=='forward' else None
    frozen_expected=expected;source_hash=hashlib.sha256(val.manifest.read_bytes()).hexdigest()
    frozen_hash=c.get('manifest_sha256',{}).get(a.split)
    if frozen_hash is not None and frozen_hash!=source_hash:expected=None
    if a.split!='val':expected=None
    if saved_metrics.get('images',len(val))!=len(val):expected=None
    report=dict(status='completed',counts=dict(counts),dataset=c['dataset'],checkpoint=str(a.checkpoint.resolve()),line_association=method,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),seconds=time.time()-started,checkpoint_epoch=s.get('next_epoch'),
        previous_full_exact=expected,previous_full_exact_reproduced=expected==counts['legacy_full_exact'] if expected is not None else None,
        split=a.split,input_manifest_rows=input_rows,unique_assets=a.unique_assets,evaluated_images=len(val),
        postprocess={k:c.get(k) for k in ['line_threshold','band_threshold','refine_line_peaks','stereo_from_band']},
        source_manifest_sha256=source_hash,frozen_checkpoint_full_exact=frozen_expected,
        checkpoint_evaluation_manifest_matches=(frozen_hash==source_hash) if frozen_hash else None)
    write(a.output/'report.json',report);write(a.output/'component_samples.json',rows);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
