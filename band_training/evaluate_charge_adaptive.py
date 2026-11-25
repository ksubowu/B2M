"""Full charged-subset development comparison, including every prior success."""
import argparse,pickle,json,time
from pathlib import Path
from collections import Counter,defaultdict
from .geometry import assemble_band_lines
from .charge_association import assign_charges
from .charge_adaptive import assign_adaptive

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);data=pickle.load(args.cache.open('rb'));rows=[];stats=defaultdict(Counter);started=time.time()
    for d in data:
        if not d['charges']:continue
        graph=assemble_band_lines(d['boxes'],d['instances'],line_association='exclusive_box')
        methods=dict(baseline=assign_charges(d['atoms'],d['charges'],graph['bonds']),
                     adaptive=assign_adaptive(d['atoms'],d['charges'],graph['bonds'],d['lines']),
                     guarded=assign_adaptive(d['atoms'],d['charges'],graph['bonds'],d['lines'],guarded=True),
                     safe_adaptive=assign_adaptive(d['atoms'],d['charges'],graph['bonds'],d['lines'],preserve_positive_hetero=True))
        truth=Counter(zip(d['reference']['charge_owners'],[q['value'] for q in d['charges']]))
        row=dict(sample_id=d['sample_id'],split=d['split'],methods={})
        for name,assignments in methods.items():
            actual=Counter((a['atom_index'],q['value']) for a,q in zip(assignments,d['charges']));okay=actual==truth
            correct=sum(a['atom_index']==gt for a,gt in zip(assignments,d['reference']['charge_owners']))
            row['methods'][name]=dict(exact=okay,correct_owners=correct,assignments=assignments)
            for group in ['all',d['split']]:stats[name+':'+group].update(images=1,exact=int(okay),charges=len(d['charges']),correct=correct)
        rows.append(row)
    comparison={}
    for name in ['adaptive','guarded','safe_adaptive']:
        changes=[r for r in rows if r['methods'][name]['exact']!=r['methods']['baseline']['exact']]
        comparison[name]=dict(corrected=[r['sample_id'] for r in changes if r['methods'][name]['exact']],
                              regressed=[r['sample_id'] for r in changes if not r['methods'][name]['exact']])
    report=dict(charged_images=len(rows),total_images=len(data),seconds=time.time()-started,stats=dict(stats),comparison=comparison,
                note='Development comparison on the same dataset, not independent test performance. Production defaults unchanged.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2));(args.output/'samples.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
