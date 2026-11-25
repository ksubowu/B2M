"""Attribute missing typed GT edges to stages using frozen dual-head outputs.

Ground truth is used for diagnosis only. No predictions are repaired.
"""
import gzip,json
from collections import Counter
from pathlib import Path
from .geometry import LineEndpointIndex,target_edges,edge_key
from .scored_bond_postprocess import assemble_scored_bonds


def main():
    root=Path('runs/scored_bond_v3_comparison_20261118')
    totals=Counter();reasons=Counter();samples=[];matrix=Counter()
    with gzip.open(root/'prediction_cache/predictions.jsonl.gz','rt') as source:
        for row in source:
            d=json.loads(row);mapping={int(k):v for k,v in d['atom_mapping'].items()}
            graph=assemble_scored_bonds([a['bbox'] for a in d['atoms']],d['independent_heads'])
            truth=target_edges(d['ann']);pred=Counter();typed_by_pair={}
            for e in graph['bonds']:
                u,v=mapping.get(e['begin'],-1),mapping.get(e['end'],-1)
                if min(u,v)<0:continue
                pred[edge_key(u,v,e['label'])]+=1
                typed_by_pair[tuple(sorted((u,v)))]=e
            line_by_pair={}
            for e in graph['line_graph']:
                u,v=mapping.get(e['begin'],-1),mapping.get(e['end'],-1)
                if min(u,v)>=0:line_by_pair[tuple(sorted((u,v)))]=e
            raw=d['independent_heads']['lines']
            raw_pairs=LineEndpointIndex([a['bbox'] for a in d['atoms']]).link_many([l['line'] for l in raw])
            raw_mapped={tuple(sorted((mapping.get(p[0],-1),mapping.get(p[1],-1)))) for p in raw_pairs if p is not None}
            gt_pairs=LineEndpointIndex([a['bbox'] for a in d['ann']['atoms']]).link_many([l['line'] for l in raw])
            gt_pair_set={tuple(sorted(p)) for p in gt_pairs if p is not None}
            available=set(mapping.values());items=[]
            for key,num in (truth-pred).items():
                pair=tuple(sorted(key[:2]));typed=typed_by_pair.get(pair)
                if typed is not None:
                    if typed['label']!=key[2]:reason='band_wrong_type_on_correct_pair'
                    else:reason='line_stereo_direction_reversed'
                elif pair in line_by_pair:
                    reason='line_correct_pair_but_no_accepted_same_query_band'
                elif not set(pair)<=available:
                    reason='atom_detection_or_class_match_missing'
                elif pair in raw_mapped:
                    reason='correct_raw_line_pair_lost_in_grouping'
                elif pair in gt_pair_set:
                    reason='line_links_correctly_with_gt_boxes_only'
                else:
                    reason='no_accepted_line_with_correct_endpoint_pair_even_with_gt_boxes'
                reasons[reason]+=num
                if typed is not None:matrix[f"{key[2]}->{typed['label']}"]+=num
                items.append(dict(edge=key,count=num,reason=reason))
            totals.update(images=1,gt=sum(truth.values()),typed_tp=sum((truth&pred).values()),typed_fn=sum((truth-pred).values()),
                untyped_line_pair_fn=sum(num for key,num in truth.items() if tuple(sorted(key[:2])) not in line_by_pair),
                line_only_records=len(graph['untyped_lines']))
            if items:samples.append(dict(sample_id=d['sample_id'],missing=items))
    assert totals['typed_fn']==468 and sum(reasons.values())==468
    report=dict(totals=dict(totals),missing_reasons=dict(reasons),class_confusion=dict(matrix),
        scope='Counts missing GT typed edges, not images. FN includes wrong class/direction and atom association failures, not only detector absence. Stage attribution uses cached thresholded line/band outputs; subthreshold raw network logits are unavailable. GT-box replacement is a diagnostic counterfactual, not a production input.')
    out=root/'missing_diagnosis';out.mkdir(exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'samples.json').write_text(json.dumps(samples,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
