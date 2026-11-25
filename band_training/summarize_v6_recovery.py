"""Refresh a self-contained comparison for the bounded recovery experiments."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;rows=[];states={}
    for arm in ['uniform_heads','replay_heads']:
        d=r/arm;pp=d/'progress.json';states[arm]=json.loads(pp.read_text()) if pp.exists() else {'status':'pending'}
        for path in sorted(d.glob('*_metrics.json')):
            if path.name=='baseline_metrics.json':epoch=0
            elif path.name.startswith('epoch_'):epoch=int(path.name.split('_')[1])
            else:continue
            m=json.loads(path.read_text());rows.append(dict(arm=arm,epoch=epoch,images=m['images'],atom_f1=m['atom_f1'],charge_f1=m['charge_f1'],bond_f1=m['typed_bond']['f1'],strict_graph=m['full_graph_exact'],graph_star=m['graph_star_exact'],graph_ocr=m['graph_ocr_exact'],rare_atom_macro_f1=m['rare_atom_macro_f1']))
    plan=json.loads((r/'experiment_plan.json').read_text());report=dict(status='completed' if all(s.get('status')=='completed' for s in states.values()) else 'running',plan=plan,progress=states,rows=rows)
    for name in ['diagnosis_epoch30','diagnosis_epoch40','calibration_epoch30','calibration_baseline']:
        path=r/name/'report.json'
        if path.exists():report[name]=json.loads(path.read_text())
    (r/'comparison.json').write_text(json.dumps(report,indent=2))
    if rows:
        with (r/'comparison.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,key in zip(axes.flat,['atom_f1','charge_f1','bond_f1','strict_graph','graph_star','graph_ocr']):
        for arm in states:
            points=sorted((x['epoch'],100*x[key]) for x in rows if x['arm']==arm)
            if points:ax.plot(*zip(*points),marker='o',label=arm)
        ax.set(title=key,xlabel='Short-run epoch (0 = original best)',ylabel='%');ax.grid(alpha=.25)
        if ax.lines:ax.legend(fontsize=8)
    fig.suptitle('Matched-budget head-only recovery / fixed development validation');fig.tight_layout();tmp=r/'comparison.tmp.png';fig.savefig(tmp,dpi=150);plt.close(fig);tmp.replace(r/'comparison.png')
    print(json.dumps(dict(status=report['status'],progress={k:s.get('status') for k,s in states.items()},metric_rows=len(rows))))


if __name__=='__main__':main()
