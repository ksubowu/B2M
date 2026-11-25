"""Join previous equal-budget controls and prior-constrained experiments."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;old=r.parent/'v6_recovery_20261124';rows=[];progress={}
    specs=[('old_uniform',old/'uniform_heads'),('old_replay',old/'replay_heads'),('prior_heads',r/'prior_heads'),('prior_bond_only',r/'prior_bond_only')]
    for arm,d in specs:
        p=d/'progress.json';progress[arm]=json.loads(p.read_text()) if p.exists() else {'status':'pending'}
        for p in sorted(d.glob('*_metrics.json')):
            if p.name=='baseline_metrics.json':epoch=0
            elif p.name.startswith('epoch_'):epoch=int(p.name.split('_')[1])
            else:continue
            m=json.loads(p.read_text());rows.append(dict(arm=arm,epoch=epoch,atom_f1=m['atom_f1'],charge_f1=m['charge_f1'],bond_f1=m['typed_bond']['f1'],strict_graph=m['full_graph_exact'],graph_star=m['graph_star_exact'],graph_ocr=m['graph_ocr_exact'],rare_atom_macro_f1=m['rare_atom_macro_f1']))
    status='completed' if all(progress[k].get('status')=='completed' for k in ['prior_heads','prior_bond_only']) else 'running'
    report=dict(status=status,plan=json.loads((r/'experiment_plan.json').read_text()),progress=progress,rows=rows,scope='Development validation only. The same source checkpoint, thresholds, OCR, validation IDs and image budget are used. Prior control selection references training data only. No automatic production promotion.')
    (r/'comparison.json').write_text(json.dumps(report,indent=2))
    if rows:
        with (r/'comparison.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,key in zip(axes.flat,['atom_f1','charge_f1','bond_f1','strict_graph','graph_star','graph_ocr']):
        for arm,d in specs:
            points=sorted((x['epoch'],100*x[key]) for x in rows if x['arm']==arm)
            if points:ax.plot(*zip(*points),marker='o',label=arm)
        ax.set(title=key,xlabel='Short experiment epoch (0 = original best)',ylabel='%');ax.grid(alpha=.25)
        if ax.lines:ax.legend(fontsize=7)
    fig.suptitle('Training-prior controlled recovery / identical development validation');fig.tight_layout();tmp=r/'comparison.tmp.png';fig.savefig(tmp,dpi=150);plt.close(fig);tmp.replace(r/'comparison.png')
    print(json.dumps(dict(status=status,progress={k:v.get('status') for k,v in progress.items()})),flush=True)


if __name__=='__main__':main()
