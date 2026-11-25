"""User-facing metrics: only true aliases/R-sites become dummy atoms.

Uses frozen graphs and already recognized strings; never reruns or retunes OCR.
Unsupported true elements retain their identity and cannot earn dummy credit.
"""
import argparse,gzip,json
from pathlib import Path
from collections import Counter,defaultdict
import networkx as nx
from .molecular_graph_metrics import exact_property_match
from .continue_indigo import write


def functional_reference(graph,labels,with_text=False):
    result=graph.copy()
    for label in labels:
        if label['kind']=='unsupported_element':
            result.nodes[label['atom_id']]['token']=label['label']
            if with_text:result.nodes[label['atom_id']]['text']=''
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--references',type=Path,required=True)
    a=p.parse_args();refs={r['sample_id']:r for r in json.loads(a.references.read_text())}
    old={r['sample_id']:r for r in json.loads((a.evaluation/'samples.json').read_text())}
    groups=defaultdict(Counter);rows=[]
    with gzip.open(a.evaluation/'graphs.jsonl.gz','rt',encoding='utf-8') as source:
        for line in source:
            d=json.loads(line);sid=d['sample_id'];before=old[sid]
            pred=nx.node_link_graph(d['predicted'],edges='edges');ref=nx.node_link_graph(d['reference'],edges='edges')
            ocr_pred=nx.node_link_graph(d['predicted_ocr'],edges='edges');ocr_ref=nx.node_link_graph(d['reference_ocr'],edges='edges')
            labels=refs[sid]['dummy_labels'];alias=[r for r in labels if r['kind']=='alias_or_rsite'];rare=[r for r in labels if r['kind']=='unsupported_element']
            ref=functional_reference(ref,labels)
            ocr_ref=functional_reference(ocr_ref,labels,with_text=True)
            star=exact_property_match(pred,ref) if before['valid_assembled_graph'] else None
            text=exact_property_match(ocr_pred,ocr_ref,with_text=True) if star is not None else None
            assert (star is not None)==before['alias_only_star_exact']
            for name in ['all','with_alias' if alias else 'without_alias']+(['with_unsupported_element'] if rare else []):
                groups[name].update(images=1,graph_star_exact=int(star is not None),graph_ocr_exact=int(text is not None))
            rows.append(dict(sample_id=sid,graph_star_exact=star is not None,graph_ocr_exact=text is not None,has_alias=bool(alias),
                has_unsupported_element=bool(rare),star_mapping=star,ocr_mapping=text,alias_reference_labels={r['atom_id']:r['label'] for r in alias}))
    result=dict(schema='functional_dummy_graph_metrics_v1',policy='Only true pseudoatom/functional-group/R-site labels are anonymous dummy nodes. True unsupported elements retain identity. OCR adds literal group labels only; other model errors are not repaired.',groups={k:dict(v) for k,v in groups.items()},
        source='Same frozen predictions, references and selected OCR output as the model-vocabulary diagnostic; no OCR rerun or candidate reselection.')
    write(a.evaluation/'functional_report.json',result);write(a.evaluation/'functional_samples.json',rows);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
