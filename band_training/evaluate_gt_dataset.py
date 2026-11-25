"""Full unique-image GT assembly audit with frozen band-line/charge algorithms."""
import argparse,csv,gzip,hashlib,json,time,shutil
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import torch
from .mixed_data import prepare_annotation,letterbox_transform
from .geometry import assemble_band_lines,edge_key,target_edges
from .charge_association import assign_charges


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=12)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.time()
    source=args.dataset;records=json.loads((source/'data_manifest.json').read_text())
    unique_train=json.loads((source/'unique_train.json').read_text());val=json.loads((source/'val.json').read_text())
    train_assets={r['annotation'] for r in unique_train};val_assets={r['annotation'] for r in val}
    assert not train_assets&val_assets
    assert len(records)==len({r['annotation'] for r in records})==len(train_assets|val_assets)
    assert {r['annotation'] for r in records}==train_assets|val_assets
    manifest_hashes={name:digest(source/name) for name in ['data_manifest.json','unique_train.json','val.json']}
    code=args.output/'source';code.mkdir()
    for filename in ['geometry.py','charge_association.py','mixed_data.py','evaluate_gt_dataset.py']:
        shutil.copyfile(Path(__file__).with_name(filename),code/filename)
    code_hashes={f.name:digest(f) for f in code.iterdir()}
    stats=defaultdict(Counter);failures=[];samples=[];all_input=hashlib.sha256();max_line_error=0.
    def load(record):
        payloads=[(source/record[key]).read_bytes() for key in ['annotation','band_target','line_target']]
        return record,[json.loads(b) for b in payloads],hashlib.sha256(b''.join(payloads)).hexdigest()
    def grade(pairs,labels,truth):
        edges=Counter(edge_key(*pair,label) for pair,label in zip(pairs,labels) if pair is not None)
        hits=sum((edges&truth).values());unlinked=sum(pair is None for pair in pairs)
        return int(edges==truth and not unlinked),hits,unlinked
    def charge_grade(assignment,charges):
        expected=Counter((q['atom_index'],q['formal_charge']) for q in charges)
        actual=Counter((a['atom_index'],q['formal_charge']) for a,q in zip(assignment,charges))
        return int(actual==expected),sum(a['atom_index']==q['atom_index'] for a,q in zip(assignment,charges))
    def report():
        summaries={}
        for key,s in stats.items():
            v=dict(s);n=s['images'];b=s['bonds'];q=s['charges'];cq=s['charged_images']
            v.update(bond_exact_rate=s['bond_exact']/max(1,n),bond_f1=s['bond_tp']/max(1,b),
                full_graph_exact_rate=s['full_exact']/max(1,n),charge_exact_rate=s['charge_exact']/max(1,n),
                charged_image_exact_rate=s['charged_exact']/max(1,cq),charge_owner_accuracy=s['charge_tp']/max(1,q),
                atom_anchor_bond_exact_rate=s['anchor_bond_exact']/max(1,n),
                legacy_full_exact_rate=s['legacy_full_exact']/max(1,n))
            summaries[key]=v
        return dict(status='completed' if len(samples)==len(records) else 'running',dataset=str(source.resolve()),
            unique_images_expected=len(records),images_processed=len(samples),seconds=round(time.time()-started,1),
            method='band_line: line endpoints -> KD-tree over BOX centers; band provides class/region',
            charge_association='joint_valence: local geometry, exclusive assignment, soft local valence',
            diagnostic_method='atom_anchor: separate coordinate-only KD-tree over supplied atoms[].center, NOT source/target IDs',
            no_ground_truth_bond_ids_used_for_assembly=True,manifest_sha256=manifest_hashes,code_sha256=code_hashes,
            input_annotation_target_stream_sha256=all_input.hexdigest(),max_line_to_true_atom_anchor_error_px=max_line_error,
            groups=summaries,failed_images=len(failures))
    with gzip.open(args.output/'assembled_graphs.jsonl.gz','wt',encoding='utf-8',compresslevel=3) as graphs,ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Bounded read-ahead avoids loading the entire corpus into memory.
        for start in range(0,len(records),192):
            for record,data,input_hash in pool.map(load,records[start:start+192]):
                original,band_target,line_target=data
                assert band_target['labels']==line_target['labels']
                assert band_target['size']==line_target['size']==[original['width'],original['height']]
                assert original['line_endpoint_mode']=='atom_center'
                assert np.allclose(band_target['quads'],[b['quad'] for b in original['bands']],atol=.001)
                assert np.allclose(line_target['lines'],[v['endpoints'] for v in original['lines']],atol=.001)
                ann,_,labels,quads=prepare_annotation(original,band_target)
                scale,offset,_=letterbox_transform(original['width'],original['height'])
                lines=(np.asarray(line_target['lines'],float)*scale+offset).tolist()
                assert len(lines)==len(quads)==len(ann['bonds'])
                boxes=[a['bbox'] for a in ann['atoms']]
                instances=[dict(label=l,quad=q,line=line) for l,q,line in zip(labels,quads,lines)]
                assembled=assemble_band_lines(boxes,instances)
                pairs=[None]*len(lines)
                for edge in assembled['bonds']:pairs[edge['bond_instance']]=(edge['begin'],edge['end'])
                atoms=[dict(symbol=a['symbol'],bbox=a['bbox']) for a in ann['atoms']]
                charge_inputs=[dict(value=q['formal_charge'],bbox=q['bbox']) for q in ann['charges']]
                joint=assign_charges(atoms,charge_inputs,assembled['bonds'],method='joint_valence')
                legacy=assign_charges(atoms,charge_inputs,assembled['bonds'],method='legacy')
                # GT references are consumed only below, after assembly is done.
                truth=target_edges(ann);bond_ok,tp,unlinked=grade(pairs,labels,truth)
                charge_ok,charge_tp=charge_grade(joint,ann['charges']);legacy_ok,_=charge_grade(legacy,ann['charges'])
                anchors=np.asarray([a['center'] for a in ann['atoms']],float)
                _,anchor_indices=cKDTree(anchors).query(np.asarray(lines).reshape(-1,2),k=1)
                anchor_pairs=anchor_indices.reshape(-1,2).tolist()
                anchor_ok,anchor_tp,_=grade(anchor_pairs,labels,truth)
                expected_ends=np.asarray([[anchors[b['begin']],anchors[b['end']]] for b in ann['bonds']])
                line_error=float(np.max(np.abs(np.asarray(lines)-expected_ends),initial=0));max_line_error=max(max_line_error,line_error)
                assert line_error<.01,'Line annotations do not follow the documented physical atom anchors'
                full_ok=bool(bond_ok and charge_ok);split='train' if record['annotation'] in train_assets else 'val'
                status=dict(sample_id=record['sample_id'],split=split,variant=record['variant'],dataset=record['dataset'],
                    image_id=record['image_id'],atoms=len(atoms),bonds=len(lines),charges=len(joint),
                    bonds_correct=tp,bond_exact=bool(bond_ok),charge_exact=bool(charge_ok),charge_correct=charge_tp,
                    full_exact=full_ok,atom_anchor_bond_exact=bool(anchor_ok),legacy_charge_exact=bool(legacy_ok),unlinked=unlinked,
                    annotation=record['annotation'],image=record['image'],input_sha256=input_hash)
                samples.append(status);all_input.update(bytes.fromhex(input_hash))
                sums=dict(images=1,atoms=len(atoms),bonds=len(lines),charges=len(joint),charged_images=int(bool(joint)),
                    bond_tp=tp,bond_exact=bond_ok,charge_tp=charge_tp,charge_exact=charge_ok,charged_exact=int(bool(joint) and charge_ok),
                    full_exact=int(full_ok),unlinked=unlinked,anchor_bond_exact=anchor_ok,anchor_bond_tp=anchor_tp,
                    legacy_charge_exact=legacy_ok,legacy_full_exact=int(bool(bond_ok and legacy_ok)),
                    charge_corrected_images=int(not legacy_ok and charge_ok),charge_regressed_images=int(legacy_ok and not charge_ok))
                for group in ['all',split,'variant:'+record['variant'],'dataset:'+record['dataset']]:stats[group].update(sums)
                formal=[0]*len(atoms)
                for q,owner in zip(charge_inputs,joint):
                    if owner['atom_index'] is not None:formal[owner['atom_index']]+=q['value']
                graph=dict(sample_id=record['sample_id'],split=split,coordinate_system='640px letterbox',
                    atoms=[dict(index=i,symbol=a['symbol'],bbox=a['bbox'],formal_charge=formal[i]) for i,a in enumerate(atoms)],
                    bonds=assembled['bonds'],unlinked_band_indices=assembled['unlinked_band_indices'],
                    charges=[dict(value=q['value'],bbox=q['bbox'],atom_index=o['atom_index']) for q,o in zip(charge_inputs,joint)])
                graphs.write(json.dumps(graph,ensure_ascii=False,separators=(',',':'))+'\n')
                if not full_ok:
                    failed_bonds=[];failed_charges=[]
                    for i,(pair,bond,label) in enumerate(zip(pairs,ann['bonds'],labels)):
                        if pair is None or edge_key(*pair,label)!=edge_key(bond['begin'],bond['end'],label):
                            failed_bonds.append(dict(index=i,predicted=pair,expected=[bond['begin'],bond['end']],label=label,
                                line=lines[i],quad=quads[i],atom_anchor_pair=anchor_pairs[i]))
                    for i,(q,o) in enumerate(zip(ann['charges'],joint)):
                        if o['atom_index']!=q['atom_index']:
                            failed_charges.append(dict(index=i,value=q['formal_charge'],predicted=o['atom_index'],expected=q['atom_index'],
                                bbox=q['bbox'],decision=o,legacy=legacy[i]))
                    failures.append(dict(**status,failed_bonds=failed_bonds,failed_charges=failed_charges))
                if len(samples)%500==0:
                    progress=report();(args.output/'progress.json').write_text(json.dumps(progress,indent=2))
                    print('EVALUATED',len(samples),'/',len(records),'seconds',round(time.time()-started,1),'failed_images',len(failures),flush=True)
    final=report();(args.output/'report.json').write_text(json.dumps(final,indent=2,ensure_ascii=False))
    (args.output/'samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2))
    (args.output/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2))
    with (args.output/'samples.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
    with (args.output/'failures.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0]),extrasaction='ignore');writer.writeheader();writer.writerows(failures)
    print(json.dumps(final,indent=2),flush=True)

if __name__=='__main__':main()
