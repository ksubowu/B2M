"""Summarize the fixed held-out evaluation without retuning the model."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .geometry import pred_edges,target_edges,atom_mapping
from .train import decode_atoms
from .predict import ATOM_CLASSES,BOND_CLASSES
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    results=json.loads((args.run/'results.json').read_text());config=json.loads((args.run/'config.json').read_text())
    history=[json.loads(line) for line in (args.run/'history.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for stage in ['tiny_overfit','band_train']:
        rows=[r for r in history if r['stage']==stage and 'total' in r]
        axes[0].plot([r['step'] for r in rows],[r['total'] for r in rows],label=stage)
    axes[0].set(xlabel='Training step',ylabel='Loss',yscale='log',title='Band loss');axes[0].legend()
    rows=[r for r in history if 'calibration_f1' in r]
    axes[1].plot([r['step'] for r in rows],[r['calibration_f1']*100 for r in rows],marker='o')
    axes[1].set(xlabel='Training step',ylabel='F1 (%)',title='Calibration: bands + true atoms',ylim=(90,100))
    vals=[results['test']['oracle_atoms']['f1'],results['test_before_atom_finetune']['predicted_atoms']['f1'],results['test']['predicted_atoms']['f1']]
    axes[2].bar(['True atoms','Old atom head','Adapted atom head'],np.array(vals)*100,color=['#39855b','#b68444','#477ac2'])
    axes[2].set(ylabel='Bond connectivity F1 (%)',ylim=(80,100),title='Untouched test set')
    axes[2].tick_params(axis='x',labelrotation=15)
    for i,v in enumerate(vals):axes[2].text(i,v*100+.3,f'{v*100:.2f}',ha='center')
    fig.tight_layout();fig.savefig(args.run/'learning_curves.png',dpi=180);plt.close(fig)
    definitions={
        'oracle_atoms':'Predicted bands geometrically connected to ground-truth atom boxes; typed, directed bond edges.',
        'predicted_atoms':'Predicted bands connected to predicted atom boxes; matched atom identities used only for evaluation.',
        'graph_exact':'All typed/directed bond edges correct, with no missing/extra/unlinked predicted bands; does not require every atom/charge detection correct.',
        'full_graph_exact':'Additionally requires all atom detections/classes and all charges attached to the correct atom. Not a SMILES or absolute R/S/E/Z exactness metric.',
        'charge_graph_exact':'All charge signs/magnitudes and geometrically attached atom identities correct; includes uncharged molecules.',
        'test_policy':'Model step and band score threshold chosen only on calibration split. Final test did not guide training or threshold selection.'}
    (args.run/'metric_definitions.json').write_text(json.dumps(definitions,indent=2))
    records=torch.load(args.run/'feature_cache.pt',map_location='cpu',weights_only=False)
    predictions=json.loads((args.run/'test_predictions.json').read_text())
    failures=[];oracle_counts=Counter();directions=Counter();types=Counter()
    for r in records:
        if r['row']['split']!='test':continue
        from .geometry import suppress
        ps=suppress([p for p in predictions[r['row']['image']] if p['score']>=results['threshold']])
        truth=target_edges(r['ann']);pred,unlinked=pred_edges(ps,[a['bbox'] for a in r['ann']['atoms']])
        missing=truth-pred;extra=pred-truth
        for (a,b,label),count in missing.items():oracle_counts[BOND_CLASSES[label]]+=count
        if missing or extra:
            failures.append(dict(image=r['row']['image'],missing=[list(k)+[v] for k,v in missing.items()],
                                 extra=[list(k)+[v] for k,v in extra.items()],unlinked=unlinked))
    analysis={'test_images':len(predictions),'oracle_atom_missing_bonds_by_type':dict(oracle_counts),
              'oracle_atom_failed_images':failures,
              'important':'Diagnostic only. No training, threshold or postprocessing changes were made based on these test failures.'}
    (args.run/'failure_analysis.json').write_text(json.dumps(analysis,indent=2))
    sources={}
    for f in Path(__file__).parent.glob('*.py'):sources[str(f.name)]=hashlib.sha256(f.read_bytes()).hexdigest()
    (args.run/'provenance.json').write_text(json.dumps({'source_sha256':sources,'torch':torch.__version__,
        'manifest_sha256':hashlib.sha256((Path(config['dataset'])/'manifest.json').read_bytes()).hexdigest()},indent=2))
    text=(f"Independent band pilot\n\nDataset: 320 unique molecules; 240 train / 40 calibration / 40 test.\n"
          f"Band training: ordered tapered quadrilateral region + existence/class center focal loss. No atom-index loss.\n"
          f"Backbone/encoder frozen. Band head trained 1800 steps; checkpoint {results['best_band_step']} selected on calibration.\n"
          f"Original atom/charge decoder independently fine-tuned 300 steps.\n\n"
          f"Test bands + true atoms: F1 {results['test']['oracle_atoms']['f1']:.4%}, bond graph exact {results['test']['oracle_atoms']['graph_exact']:.2%}.\n"
          f"Test bands + predicted atoms: F1 {results['test']['predicted_atoms']['f1']:.4%}, bond graph exact {results['test']['predicted_atoms']['graph_exact']:.2%}.\n"
          f"Atom precision/recall: {results['test']['atom_precision']:.2%} / {results['test']['atom_recall']:.2%}.\n"
          f"All atom/charge/bond graph exact: {results['test']['full_graph_exact']:.2%}.\n\n"
          f"Conclusion: supports feasibility on this small synthetic domain; not evidence of real-image generalization.\n"
          f"Charge detection/attachment, rare wedge/dash/triple examples, and residual false/missed bonds remain bottlenecks.\n"
          f"No SMILES, absolute stereochemistry or real-image benchmark has been evaluated.\n")
    (args.run/'report.txt').write_text(text)
    print(text)

if __name__=='__main__':main()
