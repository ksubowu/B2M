"""Full exact graph comparison, replaying one OCR pass under declared policies."""
import argparse,csv,json,shutil,subprocess,sys,zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('runs/scored_bond_improvements_20261119/selected/prediction_cache/predictions.jsonl.gz'))
    p.add_argument('--easyocr',type=Path,default=Path('runs/scored_bond_improvements_20261119/selected/graph_ocr_verified'))
    p.add_argument('--references',type=Path,default=Path('runs/scored_bond_v3_comparison_20261118/reference_verified/samples.json'))
    p.add_argument('--dataset',default='/mnt/hdd2/bo/work/cvs/chemcv_data/real/indigo_full_v3_20261116');a=p.parse_args()
    cache=a.cache;easy=a.easyocr;refs=a.references;dataset=a.dataset
    sources={'easyocr':easy}
    for policy in ['en_single','en_multiscale','latin_multiscale','ensemble_multiscale','en_consensus','latin_consensus','ensemble_consensus']:
        out=a.root/policy;sources[policy]=out
        if not (out/'report.json').exists():
            subprocess.run([sys.executable,'-m','band_training.evaluate_graph_ocr','--cache',str(cache),'--dataset',dataset,
                            '--output',str(out),'--paddle-results',str(a.root/'recognition/candidates.json'),'--paddle-policy',policy],check=True)
        subprocess.run([sys.executable,'-m','band_training.rescore_functional_groups','--evaluation',str(out),'--references',str(refs)],check=True)
    references=json.loads(refs.read_text());aliases={(r['sample_id'],q['atom_id']) for r in references for q in r['dummy_labels'] if q['kind']=='alias_or_rsite'}
    table=[];samples={};label_comparisons={}
    for name,out in sources.items():
        formal=json.loads((out/'functional_report.json').read_text());group=formal['groups']['all'];alias=formal['groups']['with_alias']
        crops=json.loads((out/'ocr_crops.json').read_text())
        predicted=[r for r in crops if r['kind']=='predicted_crop' and (r['sample_id'],r['reference_atom']) in aliases]
        gt=[r for r in crops if r['kind']=='gt_crop_diagnostic' and (r['sample_id'],r['reference_atom']) in aliases]
        table.append(dict(backend=name,images=group['images'],graph_star_exact=group['graph_star_exact'],graph_ocr_exact=group['graph_ocr_exact'],
            graph_ocr_rate=group['graph_ocr_exact']/group['images'],alias_images=alias['images'],alias_graph_star_exact=alias['graph_star_exact'],
            alias_graph_ocr_exact=alias['graph_ocr_exact'],matched_alias_crops=len(predicted),matched_alias_text_exact=sum(r['exact'] for r in predicted),
            gt_alias_crops=len(gt),gt_alias_text_exact=sum(r['exact'] for r in gt)))
        samples[name]={r['sample_id']:r for r in json.loads((out/'functional_samples.json').read_text())}
        label_comparisons[name]={(r['sample_id'],r['predicted_atom']):r for r in predicted}
    assert len({r['graph_star_exact'] for r in table})==1
    transitions={name:dict(fixed=[sid for sid,r in scores.items() if r['graph_ocr_exact'] and not samples['easyocr'][sid]['graph_ocr_exact']],
                           regressed=[sid for sid,r in scores.items() if not r['graph_ocr_exact'] and samples['easyocr'][sid]['graph_ocr_exact']]) for name,scores in samples.items() if name!='easyocr'}
    labels=[]
    for key,e in label_comparisons['easyocr'].items():
        row=dict(sample_id=key[0],predicted_atom=key[1],expected=e['expected'])
        for name,records in label_comparisons.items():
            r=records[key];row[name]=dict(text=r['text'],exact=r['exact'],confidence=r['confidence'])
        labels.append(row)
    review=a.root/'review';review.mkdir(exist_ok=True)
    recommended=dict(backend='PaddleOCR',policy='en_consensus',recognizer='en PP-OCRv4',scales=[1,2,3,4],margins=[1,3],
        selection='majority literal normalized string; mean confidence breaks ties',uses_gt_labels=False,uses_lexicon=False,
        ocr_python=str(Path('.venvs/paddleocr311/bin/python').absolute()),
        validation_development_only=True,changes_graph_geometry_or_topology=False)
    report=dict(status='completed',metrics=table,transitions=transitions,
        scope='Same frozen v3 weights, predicted boxes, graph topology and 1498 validation images; only literal OCR labels differ. No GT text/lexicon sent to recognizers. Official functional-only dummy policy. GT crop results are diagnostics, separate from prediction metrics.',
        candidate_selection='Declared models/scales; multiscale policies select highest nonempty confidence. Consensus policies select majority literal normalized text, breaking ties by mean confidence. No selection using expected labels. Ensemble confidence scores across models may be uncalibrated.',
        normalization='NFKC and whitespace only; case, punctuation and R indices preserved.',
        source_cache=str(cache),paddle=json.loads((a.root/'recognition/report.json').read_text()),recommended_config=recommended)
    (review/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(review/'label_comparison.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2))
    (review/'recommended_ocr.json').write_text(json.dumps(recommended,indent=2))
    with (review/'label_comparison.csv').open('w',newline='') as f:
        fields=['sample_id','predicted_atom','expected']+[name+suffix for name in sources for suffix in ['_text','_confidence','_exact']]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in labels:
            flat={k:row[k] for k in fields[:3]}
            for name in sources:
                for suffix in ['text','confidence','exact']:flat[name+'_'+suffix]=row[name][suffix]
            w.writerow(flat)
    with (review/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['OCR comparison: same 1498 frozen predicted graphs.','']
    for r in table:
        lines.append(f"{r['backend']}: graph_ocr {r['graph_ocr_exact']}/1498 = {100*r['graph_ocr_rate']:.2f}%; alias graphs {r['alias_graph_ocr_exact']}/{r['alias_graph_star_exact']}; matched label crops {r['matched_alias_text_exact']}/{r['matched_alias_crops']}; GT label crops {r['gt_alias_text_exact']}/{r['gt_alias_crops']}")
    lines+=['',report['scope'],report['candidate_selection']]
    (review/'report.txt').write_text('\n'.join(lines)+'\n')
    source=review/'source';source.mkdir(exist_ok=True)
    for name in ['evaluate_graph_ocr.py','prepare_paddle_ocr_requests.py','paddle_multiscale_worker.py','paddle_ocr_replay.py','compare_paddle_ocr.py','rescore_functional_groups.py','test_paddle_ocr_replay.py']:
        shutil.copyfile(Path(__file__).with_name(name),source/name)
    shutil.copyfile('requirements_paddle_ocr.txt',source/'requirements_paddle_ocr.txt')
    shutil.copyfile('scripts/evaluate_paddle_graph_ocr.sh',source/'evaluate_paddle_graph_ocr.sh')
    if (a.root/'environment.txt').exists():shutil.copyfile(a.root/'environment.txt',review/'environment.txt')
    with zipfile.ZipFile(a.root/'review.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in review.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(a.root))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
