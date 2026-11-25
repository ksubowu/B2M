"""Compare forward NN, proposed reverse NN, and exclusive endpoint assignments."""
import argparse,csv,gzip,hashlib,json,time,shutil
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .mixed_data import prepare_annotation,letterbox_transform
from .geometry import assemble_band_lines,edge_key,target_edges
from .charge_association import assign_charges

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);started=time.time();root=args.dataset
    records=json.loads((root/'data_manifest.json').read_text());methods=['forward','reverse_nearest','global_center','exclusive_box']
    stats={m:defaultdict(Counter) for m in methods};topology=Counter();samples=[];failures=[];stream=hashlib.sha256()
    code=args.output/'source';code.mkdir()
    for name in ['geometry.py','endpoint_assignment.py','charge_association.py','mixed_data.py','evaluate_endpoint_assignment.py']:
        shutil.copyfile(Path(__file__).with_name(name),code/name)
    def load(record):
        payloads=[(root/record[key]).read_bytes() for key in ['annotation','band_target','line_target']]
        return record,[json.loads(v) for v in payloads],hashlib.sha256(b''.join(payloads)).hexdigest()
    def report():
        results={}
        for method,groups in stats.items():
            results[method]={}
            for name,s in groups.items():
                results[method][name]=dict(s,bond_exact_rate=s['bond_exact']/max(1,s['images']),
                    bond_f1=s['bond_tp']/max(1,s['bonds']),full_exact_rate=s['full_exact']/max(1,s['images']),
                    charge_owner_accuracy=s['charge_tp']/max(1,s['charges']))
        return dict(status='completed' if len(samples)==len(records) else 'running',images_processed=len(samples),
            seconds=round(time.time()-started,1),dataset=str(root.resolve()),methods=results,topology=dict(topology),
            input_stream_sha256=stream.hexdigest(),input_manifest_sha256=sha(root/'data_manifest.json'),
            code_sha256={f.name:sha(f) for f in code.iterdir()},
            inputs='Only atom box geometry, ordered line coordinates and band classes. No supplied atom centers, GT identities, GT degrees or owner fields passed into assembly.',
            endpoint_deduplication='Exact unique coordinates on this GT corpus; prediction noise clustering not assumed.')
    with gzip.open(args.output/'assembled_graphs.jsonl.gz','wt',encoding='utf-8',compresslevel=3) as graphout,ThreadPoolExecutor(max_workers=12) as pool:
        for start in range(0,len(records),192):
            for record,data,digest in pool.map(load,records[start:start+192]):
                original,bt,lt=data;assert bt['labels']==lt['labels']
                ann,_,labels,quads=prepare_annotation(original,bt);scale,offset,_=letterbox_transform(original['width'],original['height'])
                lines=(np.asarray(lt['lines'],float)*scale+offset).tolist();boxes=[a['bbox'] for a in ann['atoms']]
                instances=[dict(label=l,quad=q,line=line) for l,q,line in zip(labels,quads,lines)]
                atoms=[dict(symbol=a['symbol'],bbox=a['bbox']) for a in ann['atoms']]
                charges=[dict(value=q['formal_charge'],bbox=q['bbox']) for q in ann['charges']]
                assembled={m:assemble_band_lines(boxes,instances,line_association=m) for m in methods}
                assigned={m:assign_charges(atoms,charges,assembled[m]['bonds']) for m in methods}
                # Scoring references begin here, never supplied to matchers.
                truth=target_edges(ann);expected_charges=Counter((q['atom_index'],q['formal_charge']) for q in ann['charges'])
                incident={i for b in ann['bonds'] for i in [b['begin'],b['end']]};isolated=len(atoms)-len(incident)
                raw=assembled['reverse_nearest']['endpoint_assignment_audit'];npoints=raw['unique_endpoints']
                topology.update(images=1,atoms=len(atoms),unique_endpoints=npoints,isolated_atoms=isolated,
                    images_with_isolated_atoms=int(isolated>0),count_equal_images=int(npoints==len(atoms)),
                    count_matches_incident_atoms_images=int(npoints==len(incident)),
                    raw_nn_collision_images=int(raw['raw_nearest_collision_endpoints']>0),
                    raw_nn_collision_endpoints=raw['raw_nearest_collision_endpoints'],
                    raw_nn_uncovered_endpoints=raw['raw_nearest_uncovered_endpoints'])
                status=dict(sample_id=record['sample_id'],split=record['split'],variant=record['variant'],dataset=record['dataset'],
                    annotation=record['annotation'],image=record['image'],atoms=len(atoms),unique_endpoints=npoints,isolated_atoms=isolated,
                    raw_collision_endpoints=raw['raw_nearest_collision_endpoints'],raw_uncovered_endpoints=raw['raw_nearest_uncovered_endpoints'],methods={})
                graph=dict(sample_id=record['sample_id'],atom_boxes=boxes,atom_symbols=[a['symbol'] for a in atoms],methods={})
                for method,a in assembled.items():
                    edges=Counter(edge_key(e['begin'],e['end'],e['label']) for e in a['bonds']);tp=sum((edges&truth).values())
                    bond_ok=int(edges==truth and not a['unlinked_band_indices'])
                    owners=assigned[method];actual=Counter((o['atom_index'],q['value']) for o,q in zip(owners,charges))
                    charge_ok=int(actual==expected_charges);full_ok=bond_ok and charge_ok
                    qtp=sum(o['atom_index']==q['atom_index'] for o,q in zip(owners,ann['charges']))
                    s=dict(images=1,atoms=len(atoms),bonds=len(lines),charges=len(charges),bond_tp=tp,
                        bond_exact=bond_ok,charge_exact=charge_ok,charge_tp=qtp,full_exact=int(full_ok),unlinked=len(a['unlinked_band_indices']))
                    for group in ['all',record['split'],'variant:'+record['variant']]:stats[method][group].update(s)
                    status['methods'][method]=dict(bond_exact=bool(bond_ok),full_exact=bool(full_ok),charge_exact=bool(charge_ok),bond_tp=tp,charge_tp=qtp)
                    graph['methods'][method]=dict(bonds=a['bonds'],unlinked_band_indices=a['unlinked_band_indices'],charges=owners)
                    if not full_ok:
                        pairs=[None]*len(lines)
                        for e in a['bonds']:pairs[e['bond_instance']]=(e['begin'],e['end'])
                        bad=[]
                        for i,(pair,b,l) in enumerate(zip(pairs,ann['bonds'],labels)):
                            if pair is None or edge_key(*pair,l)!=edge_key(b['begin'],b['end'],l):
                                bad.append(dict(band_index=i,predicted=pair,expected=[b['begin'],b['end']],label=l,line=lines[i]))
                        failures.append(dict(sample_id=record['sample_id'],split=record['split'],variant=record['variant'],
                            method=method,annotation=record['annotation'],image=record['image'],failed_bonds=bad,
                            failed_charges=[dict(index=i,predicted=o['atom_index'],expected=q['atom_index'],value=q['formal_charge']) for i,(o,q) in enumerate(zip(owners,ann['charges'])) if o['atom_index']!=q['atom_index']],
                            assignment_audit=a['endpoint_assignment_audit']))
                samples.append(status);stream.update(bytes.fromhex(digest));graphout.write(json.dumps(graph,ensure_ascii=False,separators=(',',':'))+'\n')
                if len(samples)%500==0:
                    progress=report();(args.output/'progress.json').write_text(json.dumps(progress,indent=2))
                    print('EVALUATED',len(samples),'/',len(records),'seconds',round(time.time()-started,1),
                          'bond exact',{m:stats[m]['all']['bond_exact'] for m in methods},flush=True)
    final=report();reference=json.loads(args.reference.read_text())
    assert final['input_stream_sha256']==reference['input_annotation_target_stream_sha256'],'Data changed since forward baseline'
    for group in ['all','train','val']:
        for key in ['images','atoms','bonds','charges','bond_tp','bond_exact','charge_tp','full_exact','unlinked']:
            assert final['methods']['forward'][group][key]==reference['groups'][group][key],('Baseline changed',group,key)
    final['forward_baseline_reproduced']=True
    for name,data in [('report.json',final),('samples.json',samples),('failures.json',failures)]:
        (args.output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False))
    with (args.output/'samples.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['sample_id','split','variant','atoms','unique_endpoints','isolated_atoms','raw_collision_endpoints']+[m+suffix for m in methods for suffix in ['_bond_exact','_full_exact']]
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for s in samples:
            row=dict(s)
            for m in methods:
                for suffix in ['bond_exact','full_exact']:row[m+'_'+suffix]=s['methods'][m][suffix]
            writer.writerow(row)
    print(json.dumps(final,indent=2),flush=True)

if __name__=='__main__':main()
