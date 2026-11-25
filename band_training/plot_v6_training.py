"""Offline, continuously refreshed training curves and machine-readable tables."""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def _csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def _save(fig,path):
    temp=path.with_suffix('.tmp.png');fig.savefig(temp,dpi=150,bbox_inches='tight');plt.close(fig);temp.replace(path)


def render_monitor(run):
    run=Path(run);out=run/'monitor';out.mkdir(exist_ok=True)
    config=json.loads((run/'config.json').read_text());spe=(config['train_rows']+config['batch_size']-1)//config['batch_size']
    spe=(spe+config.get('accumulation',1)-1)//config.get('accumulation',1)
    metrics=[];classes=[]
    for p in sorted(run.glob('*_metrics.json')):
        if p.name=='baseline_metrics.json':epoch=0
        elif p.name.startswith('epoch_'):epoch=int(p.name.split('_')[1])
        else:continue
        m=json.loads(p.read_text());row=dict(epoch=epoch,images=m['images'],atom_f1=m['atom_f1'],charge_f1=m['charge_f1'],bond_f1=m['typed_bond']['f1'],
            strict_graph=m['full_graph_exact'],graph_star=m['graph_star_exact'],graph_ocr=m['graph_ocr_exact'],rare_atom_macro_f1=m.get('rare_atom_macro_f1'))
        for component in ['atom','charge','bond']:
            for metric in ['precision','recall']:
                row[component+'_'+metric]=m['objects'][component][metric]
        for key in ['atom_exact','charge_detection_exact','charge_owner_exact','bond_exact']:
            if key in m['whole_images']:row[key]=m['whole_images'][key]['rate']
        for item in m.get('per_class',[]):classes.append(dict(epoch=epoch,**item))
        metrics.append(row)
    metrics.sort(key=lambda r:r['epoch']);_csv(out/'metrics_by_epoch.csv',metrics);_csv(out/'per_class_by_epoch.csv',classes)
    losses=[];hp=run/'history.jsonl'
    if hp.exists():
        for line in hp.read_text().splitlines():
            try:losses.append(json.loads(line))
            except json.JSONDecodeError:continue
    _csv(out/'loss_samples.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in losses])
    averages=[]
    for epoch in sorted({r['epoch'] for r in losses}):
        records=[r for r in losses if r['epoch']==epoch];row=dict(epoch=epoch,logged_batches=len(records),complete=any(m['epoch']==epoch for m in metrics))
        for name in ['loss','box_loss','line_loss','band_loss','charge_aux_loss','neighborhood','gradient_norm']:
            vals=[r[name] for r in records if name in r]
            if vals:row[name]=float(np.mean(vals))
        averages.append(row)
    _csv(out/'loss_by_epoch.csv',averages)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    groups=[(['atom_f1','charge_f1','bond_f1'],'Component F1'),(['strict_graph','graph_star','graph_ocr'],'Whole molecule accuracy'),
            (['atom_exact','charge_detection_exact','charge_owner_exact','bond_exact'],'All objects correct per image'),(['rare_atom_macro_f1'],'Rare atom macro F1')]
    for ax,(keys,title) in zip(axes.flat,groups):
        for key in keys:
            points=[(r['epoch'],r[key]*100) for r in metrics if r.get(key) is not None]
            if points:ax.plot(*zip(*points),marker='o',label=key)
        ax.set(title=title,xlabel='Cumulative v6 epoch (0 = original weights)',ylabel='%');ax.grid(alpha=.25)
        if ax.lines:ax.legend(fontsize=8)
        ax.axvline(2.5,color='gray',ls=':',alpha=.7)
    fig.suptitle('Fixed validation / PaddleOCR English multiscale voting');fig.tight_layout();_save(fig,out/'metrics.png')
    fig,axes=plt.subplots(3,2,figsize=(13,12))
    for ax,name in zip(axes.flat,['loss','box_loss','line_loss','band_loss','charge_aux_loss','gradient_norm']):
        points=[(r['step']/spe,r[name]) for r in losses if name in r]
        if points:
            x,y=map(np.asarray,zip(*points));ax.plot(x,y,alpha=.2,lw=.6,label='Logged every 20 updates')
            window=min(50,len(y));smooth=np.convolve(y,np.ones(window)/window,mode='valid');ax.plot(x[window-1:],smooth,lw=1.3,label='50-point rolling mean')
        ax.set(title=name,xlabel='Cumulative v6 epoch',ylabel='Loss' if name!='gradient_norm' else 'Before clipping');ax.grid(alpha=.25)
        if ax.lines:ax.legend(fontsize=8)
        ax.axvline(2.5,color='gray',ls=':',alpha=.7)
    fig.suptitle('Training batch losses (historical charge auxiliary loss unavailable)');fig.tight_layout();_save(fig,out/'loss.png')
    html='''<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>v6 training monitor</title></head><body style="font-family:sans-serif;max-width:1400px;margin:auto"><h2>v6 training monitor / 训练监控</h2><p>第 3–22 轮继续训练，关闭早停。基线为第 0 轮；图表包含原第 1–2 轮。每 200 次更新及每轮验证后刷新文件。</p><p>Loss 每 20 次更新记录；均值为记录批次的平均值，不是完整训练集 loss。页面每 60 秒刷新；下载到本地的静态文件不会同步服务器更新。</p><p><a href="metrics_by_epoch.csv">Metrics CSV</a> | <a href="loss_by_epoch.csv">Loss epoch CSV</a> | <a href="loss_samples.csv">Loss samples CSV</a></p><img src="metrics.png" style="width:100%"><img src="loss.png" style="width:100%"></body></html>'''
    html=html.replace('第 3–22 轮继续训练，关闭早停。',f"训练目标：累计到第 {config['epochs']} 轮；早停{'开启' if config.get('early_stopping_enabled') else '关闭'}。")
    (out/'index.html').write_text(html)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();render_monitor(a.run)
