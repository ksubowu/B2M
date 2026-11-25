"""Public safe_adaptive API regression on every unique GT image."""
import argparse,pickle,json,gzip,time,hashlib,shutil
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from .geometry import assemble_band_lines
from .charge_association import assign_charges

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--reference-graphs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    data=pickle.load(args.cache.open('rb'));started=time.time();stats=defaultdict(Counter);rows=[];remaining=[];fixed=[];regressed=[]
    with gzip.open(args.reference_graphs,'rt',encoding='utf-8') as reference,gzip.open(args.output/'assembled_graphs.jsonl.gz','wt',encoding='utf-8',compresslevel=3) as output:
        for d,line in zip(data,reference):
            prior=json.loads(line);assert d['sample_id']==prior['sample_id'];assert np.allclose(d['boxes'],prior['atom_boxes'],atol=1e-9)
            graph=assemble_band_lines(d['boxes'],d['instances'],line_association='exclusive_box')
            assert [(e['begin'],e['end'],e['label']) for e in graph['bonds']]==[(e['begin'],e['end'],e['label']) for e in prior['methods']['exclusive_box']['bonds']]
            old=assign_charges(d['atoms'],d['charges'],graph['bonds'])
            assert [o['atom_index'] for o in old]==[o['atom_index'] for o in prior['methods']['exclusive_box']['charges']]
            new=assign_charges(d['atoms'],d['charges'],graph['bonds'],method='safe_adaptive',lines=d['lines'])
            truth=Counter(zip(d['reference']['charge_owners'],[q['value'] for q in d['charges']]))
            baseline=Counter((a['atom_index'],q['value']) for a,q in zip(old,d['charges']))==truth
            exact=Counter((a['atom_index'],q['value']) for a,q in zip(new,d['charges']))==truth
            qtp=sum(o['atom_index']==owner for o,owner in zip(new,d['reference']['charge_owners']))
            for group in ['all',d['split']]:stats[group].update(images=1,bonds=len(graph['bonds']),charges=len(new),charged_images=int(bool(new)),
                baseline_exact=int(baseline),safe_exact=int(exact),safe_owners_correct=qtp,
                charged_exact=int(bool(new) and exact),corrected=int(exact and not baseline),regressed=int(baseline and not exact))
            row=dict(sample_id=d['sample_id'],split=d['split'],annotation=d['annotation'],image=d['image'],baseline_exact=baseline,safe_exact=exact,
                charges=len(new),baseline=old,assignments=new)
            rows.append(row)
            if not baseline and exact:fixed.append(row)
            if baseline and not exact:regressed.append(row)
            if not exact:
                remaining.append(dict(**row,errors=[dict(charge_index=i,value=q['value'],expected=expected,predicted=o['atom_index'],bbox=q['bbox']) for i,(q,expected,o) in enumerate(zip(d['charges'],d['reference']['charge_owners'],new)) if expected!=o['atom_index']]))
            formal=[0]*len(d['atoms'])
            for q,o in zip(d['charges'],new):
                if o['atom_index'] is not None:formal[o['atom_index']]+=q['value']
            output.write(json.dumps(dict(sample_id=d['sample_id'],atom_boxes=d['boxes'],atoms=[dict(index=i,symbol=a['symbol'],formal_charge=formal[i]) for i,a in enumerate(d['atoms'])],
                bonds=graph['bonds'],charges=[dict(value=q['value'],bbox=q['bbox'],atom_index=o['atom_index']) for q,o in zip(d['charges'],new)]),separators=(',',':'))+'\n')
    assert len(rows)==14980
    results={}
    for group,s in stats.items():results[group]=dict(s,baseline_rate=s['baseline_exact']/s['images'],safe_rate=s['safe_exact']/s['images'])
    report=dict(images=len(rows),seconds=time.time()-started,method='safe_adaptive',line_association='exclusive_box',
        baseline_graphs_reproduced=True,baseline_charge_owners_reproduced=True,groups=results,
        cache_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        note='Full GT development regression, not independent test or learned-model accuracy. Existing production defaults unchanged; optional safe_adaptive requires paired line geometry.')
    for name,obj in [('report.json',report),('samples.json',rows),('remaining_failures.json',remaining),('corrected_cases.json',fixed),('regressed_cases.json',regressed)]:
        (args.output/name).write_text(json.dumps(obj,indent=2))
    code=args.output/'source';code.mkdir()
    for filename in ['geometry.py','endpoint_assignment.py','charge_association.py','charge_adaptive.py','evaluate_charge_safe_full.py']:
        shutil.copyfile(Path(__file__).with_name(filename),code/filename)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
