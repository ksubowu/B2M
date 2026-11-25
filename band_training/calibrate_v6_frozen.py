"""Higher-threshold development ablations on frozen accepted endpoints only."""
import argparse,gzip,json,time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from .continue_indigo import match_boxes
from .scored_bond_postprocess import assemble_scored_bonds
from .train_scored_bonds import typed_metric
from .charge_association import assign_charges
from .train import charge_exact


def variants():
    result={'unchanged':{}}
    for t in [.35,.5]:result['atom_'+str(t)]={'atom':t}
    for t in [.5,.75]:result['rare_IPSi_'+str(t)]={'rare':t}
    for t in [.35,.5]:result['line_'+str(t)]={'line':t}
    for t in [.35,.5]:result['band_'+str(t)]={'band':t}
    result['rare50_line35']={'rare':.5,'line':.35}
    return result


def one(d):
    output={}
    for name,c in variants().items():
        atoms=[a for a in d['atoms'] if a['score']>=max(c.get('atom',.25),c.get('rare',.25) if a['label'] in [9,10,12] else .25)]
        decoded=d['independent_heads'];selected=dict(decoded,lines=[r for r in decoded['lines'] if r['line_score']>=c.get('line',.25)],bands=[r for r in decoded['bands'] if r['score']>=c.get('band',.2)],paired_band_probes=[r for r in decoded.get('paired_band_probes',[]) if r['score']>=c.get('band',.2)])
        graph=assemble_scored_bonds([a['bbox'] for a in atoms],selected,stereo_from_band=True)
        mapping=match_boxes(atoms,d['ann']['atoms']);bond=typed_metric(graph,d['ann'],mapping)
        owners=assign_charges(atoms,d['charges'],graph['bonds'],method='safe_adaptive',lines=[l['line'] for l in selected['lines']])
        charge_ok=bool(charge_exact(atoms+d['charges'],atoms,mapping,d['ann'],owners));atom_ok=len(mapping)==len(atoms)==len(d['ann']['atoms'])
        output[name]=dict(images=1,atom_tp=len(mapping),atom_pred=len(atoms),atom_gt=len(d['ann']['atoms']),bond_tp=bond['tp'],bond_fp=bond['fp'],bond_fn=bond['fn'],atom_exact=int(atom_ok),charge_owner_exact=int(charge_ok),bond_exact=int(bond['exact']),strict_exact=int(atom_ok and charge_ok and bond['exact']))
        if name=='unchanged':assert output[name]['strict_exact']==int(d['components']['legacy_full_exact']),d['sample_id']
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);totals={k:Counter() for k in variants()};started=time.time()
    with gzip.open(a.cache,'rt') as f,ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,result in enumerate(pool.map(one,map(json.loads,f),chunksize=8),1):
            for name,c in result.items():totals[name].update(c)
            if i%128==0:print('CALIBRATED',i,flush=True)
    rows=[]
    for name,c in totals.items():
        rows.append(dict(name=name,config=variants()[name],counts=dict(c),strict_rate=c['strict_exact']/c['images'],atom_f1=2*c['atom_tp']/(c['atom_pred']+c['atom_gt']),bond_f1=2*c['bond_tp']/(2*c['bond_tp']+c['bond_fp']+c['bond_fn'])))
    rows.sort(key=lambda r:(r['strict_rate'],r['bond_f1']),reverse=True)
    report=dict(status='completed',seconds=time.time()-started,source=str(a.cache),results=rows,scope='Development-only calibration; not independent test. Only higher thresholds available from accepted candidate cache; endpoint geometry is frozen. Unchanged replay must exactly reproduce every sample. No model weights or production thresholds are modified. OCR is not re-evaluated; strict component graph is the selection proxy.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
