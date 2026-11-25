"""Bounded validation-only threshold/association ablations on frozen outputs."""
import argparse,csv,gzip,json,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .geometry import assemble_band_lines,target_edges,edge_key
from .scored_bond_postprocess import assemble_scored_bonds,refine_peak_lines
from .train_scored_bonds import typed_metric
from .train import charge_exact
from .charge_association import assign_charges
from .continue_indigo import write


def variants():
    v={f'L{int(l*100):02d}_B20':dict(line=l,band=.2) for l in [.05,.1,.15,.2,.25,.3]}
    v.update({f'L20_B{int(b*100):02d}':dict(line=.2,band=b) for b in [.1,.15,.25]})
    v['no_peak_L20']=dict(line=.2,band=.2,peaks=False)
    v['no_geometry_nms_L20']=dict(line=.2,band=.2,nms=0.)
    for radius in [2.,4.,6.]:
        v[f'exclusive_noisy_r{int(radius)}']=dict(line=.2,band=.2,association='exclusive_box_noisy',radius=radius)
    v['stereo_band_L20']=dict(line=.2,band=.2,stereo_from_band=True)
    v['stereo_band_L15']=dict(line=.15,band=.2,stereo_from_band=True)
    v['stereo_band_L25']=dict(line=.25,band=.2,stereo_from_band=True)
    v['refine_peak_L20']=dict(line=.2,band=.2,refine=True)
    v['refine_peak_stereo_L20']=dict(line=.2,band=.2,refine=True,stereo_from_band=True)
    v['refine_peak_stereo_L25']=dict(line=.25,band=.2,refine=True,stereo_from_band=True)
    return v


def filtered(decoded,config):
    lines=[r for r in decoded['lines'] if r['line_score']>=config['line'] and (not config.get('peaks',True) or r['is_local_peak'])]
    queries={r['query_id'] for r in lines}
    return dict(lines=lines,bands=[r for r in decoded['bands'] if r['score']>=config['band']],
                paired_band_probes=[r for r in decoded['paired_band_probes'] if r['score']>=config['band'] and r['query_id'] in queries])


def evaluate_one(d,config):
    decoded=d['independent_heads']
    if config.get('refine'):decoded=refine_peak_lines([a['bbox'] for a in d['atoms']],decoded,config['line'])
    selected=filtered(decoded,config)
    graph=assemble_scored_bonds([a['bbox'] for a in d['atoms']],selected,line_nms_radius=config.get('nms',3.),
        line_association=config.get('association','forward'),endpoint_cluster_radius=config.get('radius',0.),stereo_from_band=config.get('stereo_from_band',False))
    mapping={int(k):v for k,v in d['atom_mapping'].items()};score=typed_metric(graph,d['ann'],mapping)
    owners=assign_charges(d['atoms'],d['charges'],graph['bonds'],method='safe_adaptive',lines=[l['line'] for l in selected['lines']])
    atom_ok=len(mapping)==len(d['atoms'])==len(d['ann']['atoms'])
    charge_ok=bool(charge_exact(d['atoms']+d['charges'],d['atoms'],mapping,d['ann'],owners))
    c=dict(atom_exact=atom_ok,charge_owner_exact=charge_ok,bond_exact=bool(score['exact']),legacy_full_exact=bool(atom_ok and charge_ok and score['exact']),
           charge_detection_exact=len(d['charge_detection_mapping'])==len(d['charges'])==len(d['ann']['charges']))
    stats=Counter(images=1,bond_tp=score['tp'],bond_fp=score['fp'],bond_fn=score['fn'],untyped_lines=len(graph['untyped_lines']))
    stats.update({k:int(v) for k,v in c.items()})
    return graph,score,owners,c,stats,selected


def process(d):
    scores={}
    for name,config in variants().items():
        _,_,_,_,stats,_=evaluate_one(d,config);scores[name]=dict(stats)
    gt={};boxes=[a['bbox'] for a in d['ann']['atoms']]
    instances=[dict(label=b['category_id']-1,line=l,quad=q) for b,l,q in zip(d['ann']['bonds'],d['gt_lines'],d['gt_quads'])]
    truth=target_edges(d['ann'])
    for method,radius in [('forward',0.),('exclusive_box',0.),('exclusive_box_noisy',2.),('exclusive_box_noisy',4.),('exclusive_box_noisy',6.)]:
        g=assemble_band_lines(boxes,instances,line_association=method,endpoint_cluster_radius=radius)
        p=Counter(edge_key(e['begin'],e['end'],e['label']) for e in g['bonds'])
        gt[f'{method}_r{radius:g}']=dict(images=1,bond_tp=sum((truth&p).values()),bond_gt=sum(truth.values()),bond_exact=int(p==truth and not g['unlinked_band_indices']))
    return dict(sample_id=d['sample_id'],variants=scores,gt=gt)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--export',choices=list(variants()));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    parent=json.loads((a.cache.parent/'report.json').read_text())
    with gzip.open(a.cache,'rt') as f:data=[json.loads(line) for line in f]
    if a.export:
        counts=Counter();samples=[]
        with gzip.open(a.output/'predictions.jsonl.gz','wt',compresslevel=3) as out:
            for d in data:
                graph,score,owners,comp,stats,selected=evaluate_one(d,variants()[a.export]);counts.update(stats)
                counts.update(atom_tp=len(d['atom_mapping']),atom_pred=len(d['atoms']),atom_gt=len(d['ann']['atoms']),
                    charge_tp=len(d['charge_detection_mapping']),charge_pred=len(d['charges']),charge_gt=len(d['ann']['charges']),charged_images=int(bool(d['ann']['charges'])))
                if d['ann']['charges']:counts.update(charged_owner_exact=int(comp['charge_owner_exact']),charged_detection_exact=int(comp['charge_detection_exact']))
                else:counts.update(neutral_images=1,neutral_charge_exact=int(comp['charge_owner_exact']))
                row={k:v for k,v in d.items() if k not in ['dense_diagnostic','gt_lines','gt_quads']}
                row.update(instances=graph['instances'],bonds=graph['bonds'],unlinked=graph['unlinked_band_indices'],untyped_lines=graph['untyped_lines'],
                           independent_heads=selected,components=comp,bond_metric=score,charge_assignments=owners)
                out.write(json.dumps(row)+'\n');samples.append(dict(sample_id=d['sample_id'],**comp))
        write(a.output/'report.json',dict(parent,counts=dict(counts),variant=a.export,variant_config=variants()[a.export],previous_full_exact_reproduced=None))
        write(a.output/'component_samples.json',samples);return
    totals={k:Counter() for k in variants()};gt_totals={};rows=[];started=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(process,data,chunksize=8):
            rows.append(result)
            for k,c in result['variants'].items():totals[k].update(c)
            for k,c in result['gt'].items():gt_totals.setdefault(k,Counter()).update(c)
            if len(rows)%128==0:print('SCORED',len(rows),'seconds',round(time.time()-started),flush=True)
    table=[];base={r['sample_id']:r['variants']['L20_B20'] for r in rows}
    for k,c in totals.items():
        tp,fp,fn=(c['bond_'+n] for n in ['tp','fp','fn']);n=c['images']
        table.append(dict(variant=k,config=variants()[k],counts=dict(c),full_exact_rate=c['legacy_full_exact']/n,
                         bond_f1=2*tp/(2*tp+fp+fn),bond_precision=tp/(tp+fp),bond_recall=tp/(tp+fn),
                         fixed=sum(not base[r['sample_id']]['legacy_full_exact'] and r['variants'][k]['legacy_full_exact'] for r in rows),
                         regressed=sum(base[r['sample_id']]['legacy_full_exact'] and not r['variants'][k]['legacy_full_exact'] for r in rows)))
    table.sort(key=lambda r:(r['counts']['legacy_full_exact'],r['bond_f1']),reverse=True)
    report=dict(status='completed',images=len(rows),seconds=time.time()-started,results=table,ground_truth_geometry={k:dict(c) for k,c in gt_totals.items()},
                source=parent,scope='Validation-only bounded ablations, not independent test estimates. No GT geometry, identity or labels enters prediction assembly. GT input experiments are separate controls. Training data and weights unchanged.')
    write(a.output/'report.json',report);write(a.output/'samples.json',rows)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
