"""Summarize a completed continuation run and export its selected inference weights."""
import argparse,json,time
from pathlib import Path
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--no-register',action='store_true');args=parser.parse_args()
    run=args.run;result=json.loads((run/'results.json').read_text());config=json.loads((run/'config.json').read_text())
    baseline=json.loads((run/'baseline_metrics.json').read_text()) if (run/'baseline_metrics.json').exists() else None
    best=result['best_metrics']
    history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for key in ['atom_charge','band']:
        axes[0].plot([r['step'] for r in history],[r[key] for r in history],label=key,alpha=.7)
    axes[0].set(xlabel='Optimizer step',ylabel='Loss',yscale='log',title='Continuation from trained weights');axes[0].legend()
    epochs=[0] if baseline else [];atoms=[baseline['atom_f1']*100] if baseline else [];bonds=[baseline['bond_graph']['f1']*100] if baseline else []
    for row in result['history']:
        epochs.append(row['epoch']);atoms.append(row['validation']['atom_f1']*100);bonds.append(row['validation']['bond_graph']['f1']*100)
    axes[1].plot(epochs,atoms,'o-',label='Atom box F1');axes[1].plot(epochs,bonds,'o-',label='Bond connectivity F1')
    charges=([baseline['charge_f1']*100] if baseline else [])+[r['validation']['charge_f1']*100 for r in result['history']]
    axes[1].plot(epochs,charges,'o-',label='Charge box F1')
    axes[1].set(xlabel='Epoch (0 = initial weights)',ylabel='Validation F1 (%)',title='Fixed original validation split');axes[1].legend()
    labels=['Atom','Charge','Band + true atoms','Band + predicted atoms']
    values=[best['atom_f1'],best['charge_f1'],best['oracle_atoms']['f1'],best['bond_graph']['f1']]
    axes[2].bar(range(4),[v*100 for v in values]);axes[2].set_xticks(range(4),labels,rotation=18)
    axes[2].set(ylabel='F1 (%)',title=f'Selected epoch {result["best_epoch"]}',ylim=(0,105))
    for i,v in enumerate(values):axes[2].text(i,v*100+1,f'{v*100:.2f}',ha='center')
    fig.tight_layout();fig.savefig(run/'learning_curves.png',dpi=170);plt.close(fig)
    state=torch.load(run/'best_joint.pt',map_location='cpu',weights_only=False)
    inference={k:state[k] for k in ['schema','base','band','epoch','step','metrics','threshold','config']}
    torch.save(inference,run/'inference_joint.pt')
    provenance=dict(schema='latest_atom_band_model_v1',run=str(run.resolve()),
                    checkpoint=str((run/'inference_joint.pt').resolve()),
                    selected_epoch=result['best_epoch'],optimizer_step=state['step'],
                    validation_bond_f1=best['bond_graph']['f1'],dataset=config['dataset'],
                    validation_charge_f1=best['charge_f1'],validation_atom_f1=best['atom_f1'],
                    source_dataset=config.get('source_dataset',config['dataset']),selection=config.get('selection','bond'),
                    webapp_weights_changed=False)
    if result['best_epoch']>0 and not args.no_register:
        registry=run.parent/'latest_atom_band_model.json'
        tmp=registry.with_suffix('.tmp');tmp.write_text(json.dumps(provenance,indent=2));tmp.replace(registry)
    text=(f"Indigo mixed dataset continuation\n\n"
          f"Training/validation images: {result['train_images']} / {result['validation_images']}.\n"
          f"Original validation split retained; trained atom/charge and band weights loaded strictly.\n"
          f"Sampling entries (including explicit charge repeats): {result['train_images']}.\n"
          f"Selection rule: {config.get('selection','bond')}.\n"
          f"RGB images and all boxes/quads use the same rounded letterbox transform to 640x640.\n"
          f"Alignment audit: {config['alignment_report']}.\n"
          f"Completed epochs: {result['epochs_completed']}; steps: {result['optimizer_steps']}; selected epoch: {result['best_epoch']}.\n\n"
          f"Initial bond connectivity F1: {baseline['bond_graph']['f1']:.4%}.\n" if baseline else '')
    text+=(f"Selected atom box F1: {best['atom_f1']:.4%}.\n"
           f"Selected charge box F1: {best['charge_f1']:.4%}; TP / predicted / GT: {best['charge_tp']} / {best['charge_pred']} / {best['charge_gt']}.\n"
           f"Selected predicted-band + true-atom connectivity F1: {best['oracle_atoms']['f1']:.4%}.\n"
           f"Selected predicted-band + predicted-atom connectivity F1: {best['bond_graph']['f1']:.4%}.\n"
           f"Bond graph exact: {best['bond_graph']['graph_exact']:.4%}; all atom/charge/bond graph exact: {best['full_graph_exact']:.4%}.\n"
           f"True-band + true-atom geometry ceiling F1: {best['geometry_ceiling']['f1']:.4%}.\n\n"
           f"Metrics use validation for checkpoint selection; they are not independent-test or real-scan generalization results.\n"
           f"Band training uses only class and ordered region coordinates, no atom index/association loss.\n"
           f"Only 20 validation images carry charge annotations; inspect charge counts, not only uncharged-image accuracy.\n"
           f"The current WebApp has NOT been switched to these weights.\n")
    (run/'report.txt').write_text(text)
    print(text,flush=True)

if __name__=='__main__':main()
