"""Explain GT audit failures and the explicitly separate atom-anchor input mode."""
import argparse,json,gzip,zipfile,html,re,copy,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw,ImageFont
from .mixed_data import prepare_annotation,letterbox_transform
from .geometry import edge_key,target_edges
from .charge_association import assign_charges


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
    report=json.loads((run/'report.json').read_text());assert report['status']=='completed'
    root=Path(report['dataset']);samples=json.loads((run/'samples.json').read_text());failures=json.loads((run/'failures.json').read_text())
    assert len(samples)==14980 and report['groups']['all']['atoms']==344768
    assert report['groups']['all']['bonds']==368318 and report['groups']['all']['charges']==3119
    newrows={s['sample_id']:copy.deepcopy(s) for s in samples};updated_graphs={};anchor_charge_changes=[]
    def load(f):
        annotation=json.loads((root/f['annotation']).read_text())
        bt=json.loads((root/annotation['band_target']).read_text());lt=json.loads((root/annotation['line_target']).read_text())
        ann,_,labels,quads=prepare_annotation(annotation,bt);scale,offset,_=letterbox_transform(annotation['width'],annotation['height'])
        lines=np.asarray(lt['lines'],float)*scale+offset
        return annotation,ann,labels,quads,lines
    # The initial audit already evaluated atom-anchor KD-tree on ALL images.
    # Only 94 images have different edge lists; re-run charge assignment there.
    # For every other image both edge multisets coincide, so charge inputs match.
    for f in failures:
        if not f['failed_bonds']:continue
        original,ann,labels,quads,lines=load(f)
        points=np.asarray([a['center'] for a in ann['atoms']]);_,indices=cKDTree(points).query(lines.reshape(-1,2),k=1)
        pairs=indices.reshape(-1,2);edges=[dict(bond_instance=i,begin=int(p[0]),end=int(p[1]),label=l) for i,(p,l) in enumerate(zip(pairs,labels))]
        assert Counter(edge_key(e['begin'],e['end'],e['label']) for e in edges)==target_edges(ann)
        atoms=[dict(symbol=a['symbol'],bbox=a['bbox']) for a in ann['atoms']]
        charges=[dict(value=q['formal_charge'],bbox=q['bbox']) for q in ann['charges']]
        assigned=assign_charges(atoms,charges,edges)
        correct=sum(q['atom_index']==a['atom_index'] for q,a in zip(ann['charges'],assigned))
        expected=Counter((q['atom_index'],q['formal_charge']) for q in ann['charges'])
        actual=Counter((a['atom_index'],q['value']) for a,q in zip(assigned,charges));charge_ok=actual==expected
        row=newrows[f['sample_id']];row.update(bond_exact=True,bonds_correct=len(edges),charge_correct=correct,
            charge_exact=bool(charge_ok),full_exact=bool(charge_ok),unlinked=0)
        if row['charge_exact']!=f['charge_exact']:anchor_charge_changes.append(dict(sample_id=f['sample_id'],old=f['charge_exact'],new=row['charge_exact']))
        formal=[0]*len(atoms)
        for q,a in zip(charges,assigned):
            if a['atom_index'] is not None:formal[a['atom_index']]+=q['value']
        updated_graphs[f['sample_id']]=dict(sample_id=f['sample_id'],split=f['split'],association='line_kdtree_supplied_atom_anchor',
            atoms=[dict(index=i,symbol=a['symbol'],bbox=a['bbox'],formal_charge=formal[i],center=points[i].tolist()) for i,a in enumerate(atoms)],
            bonds=edges,unlinked_band_indices=[],charges=[dict(value=q['value'],bbox=q['bbox'],atom_index=a['atom_index']) for q,a in zip(charges,assigned)])
    modes={}
    for group in ['all','train','val']:
        rows=[v for v in newrows.values() if group=='all' or v['split']==group];n=len(rows)
        modes[group]=dict(images=n,bond_exact_images=sum(r['bond_exact'] for r in rows),
            bonds=sum(r['bonds'] for r in rows),bonds_correct=sum(r['bonds_correct'] for r in rows),
            charge_exact_images=sum(r['charge_exact'] for r in rows),charge_correct=sum(r['charge_correct'] for r in rows),
            charges=sum(r['charges'] for r in rows),full_exact_images=sum(r['full_exact'] for r in rows),
            full_exact_rate=sum(r['full_exact'] for r in rows)/n)
    diagnostic=dict(mode='Additional supplied GT atoms[].center coordinates, not inferred box centers',
        note='All 14980 line/anchor KD queries were graded in the original audit; charges rechecked on the 94 changed-edge cases. No source/target IDs passed into association.',
        groups=modes,changed_charge_results=anchor_charge_changes,
        remaining_failures=[r for r in newrows.values() if not r['full_exact']])
    (run/'atom_anchor_input_diagnostic.json').write_text(json.dumps(diagnostic,indent=2,ensure_ascii=False))
    with gzip.open(run/'assembled_graphs_atom_anchor.jsonl.gz','wt',encoding='utf-8',compresslevel=3) as output,gzip.open(run/'assembled_graphs.jsonl.gz','rt',encoding='utf-8') as original_graphs:
        count=0
        for line in original_graphs:
            graph=json.loads(line);graph=updated_graphs.get(graph['sample_id'],graph)
            graph['association']='line_kdtree_supplied_atom_anchor';output.write(json.dumps(graph,ensure_ascii=False,separators=(',',':'))+'\n');count+=1
        assert count==14980
    (run/'failure_previews').mkdir(exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);cards=[]
    for number,f in enumerate(failures,1):
        original,ann,labels,quads,lines=load(f);im=Image.open(root/f['image']).convert('RGB')
        scale,offset,shape=letterbox_transform(*im.size);base=Image.new('RGB',(640,640),'white');base.paste(im.resize(shape,Image.Resampling.LANCZOS),tuple(offset))
        gt=base.copy();pred=base.copy();gd=ImageDraw.Draw(gt);pd=ImageDraw.Draw(pred)
        involved=set()
        for e in f['failed_bonds']:involved.update(e['expected']);involved.update(e['predicted'] or [])
        for e in f['failed_charges']:involved.add(e['expected']);involved.update([] if e['predicted'] is None else [e['predicted']])
        boxes=np.asarray([a['bbox'] for a in ann['atoms']]);centers=boxes[:,:2]+boxes[:,2:]/2
        for i in involved:
            a=ann['atoms'][i];x,y,w,h=a['bbox'];cx,cy=centers[i];ax,ay=a['center']
            for draw in [gd,pd]:
                draw.rectangle([x,y,x+w,y+h],outline='#0076c0',width=1)
                draw.text((x,max(0,y-13)),f'{i}:{a["symbol"]}',font=font,fill='#0076c0',stroke_width=1,stroke_fill='white')
                draw.ellipse([cx-2,cy-2,cx+2,cy+2],fill='red');draw.ellipse([ax-2,ay-2,ax+2,ay+2],fill='green')
        for e in f['failed_bonds']:
            q=e['quad'];poly=[tuple(v) for v in q]
            gd.line(poly+[poly[0]],fill='#888888',width=1);gd.line([tuple(v) for v in e['line']],fill='blue',width=2)
            if e['predicted'] is not None:pd.line([tuple(centers[j]) for j in e['predicted']],fill='red',width=2)
            else:pd.text(tuple(np.asarray(e['line']).mean(0)),f'B{e["index"]}:unlinked',font=font,fill='red')
        for e in f['failed_charges']:
            x,y,w,h=e['bbox'];qc=(x+w/2,y+h/2)
            for draw,owner,color in [(gd,e['expected'],'blue'),(pd,e['predicted'],'red')]:
                draw.rectangle([x,y,x+w,y+h],outline='magenta',width=2)
                if owner is not None:draw.line([qc,tuple(centers[owner])],fill=color,width=2)
                draw.text((x,max(0,y-15)),f'Q{e["index"]}:{e["value"]:+d}',font=font,fill='magenta')
        canvas=Image.new('RGB',(1920,730),'white');draw=ImageDraw.Draw(canvas)
        for col,(panel,title) in enumerate(zip([base,gt,pred],['New original image','GT line / expected charge owner','Current box-center assembly errors'])):
            canvas.paste(panel,(col*640,40));draw.text((col*640+5,8),title,font=font,fill='black')
        draw.text((5,686),f'{number:03d}/{len(failures)} | {f["split"]} | {f["sample_id"]}',font=font,fill='black')
        draw.text((5,708),'Green dots = supplied atom anchors; red dots = box centers. This preview shows the fixed BOX-center baseline.',font=font,fill='#555555')
        stem=f'{number:03d}_'+re.sub('[^a-zA-Z0-9_.-]+','_',f['sample_id'])[-160:];path='failure_previews/'+stem+'.png';canvas.save(run/path)
        cards.append(f'<article><h2>{html.escape(f["sample_id"])}</h2><p>{f["split"]}; bad bonds={len(f["failed_bonds"])}; bad charges={len(f["failed_charges"])}</p><a href="{path}"><img loading="lazy" src="{path}"></a></article>')
    lines=['Full Indigo v3 GT molecular graph assembly evaluation','',f'Dataset: {root}',
        'Primary: frozen band_line (KD-tree over atom BOX centers), joint_valence charge association. No training, neural inference, OCR or data editing.',
        'GT source/target IDs, atom charge owners and formal-charge fields are scoring references only, never consumed by association.',
        'Unique images counted once: 13482 train + 1498 validation = 14980. Explicit training sampling repeats not counted.',
        '', 'PRIMARY / BOX CENTERS:']
    for group in ['train','val','all']:
        s=report['groups'][group];lines.append(f'{group}: full graph {s["full_exact"]}/{s["images"]} ({s["full_graph_exact_rate"]:.6%}); all bonds {s["bond_exact"]}/{s["images"]} ({s["bond_exact_rate"]:.6%}); charge owners {s["charge_tp"]}/{s["charges"]}.')
    lines.extend(['',f'Primary failures: {len(failures)} images; 94 bond-only and 29 charge-only; 94 incorrect bonds, 80 unlinked.',
        'Input convention: README line endpoints are atoms[].center (drawing anchors), not necessarily box centers. Large/shifted pseudoatom labels can make BOX-center nearest neighbors incorrect.',
        '', 'SEPARATE ADDITIONAL-INPUT DIAGNOSTIC / SUPPLIED GT ATOM ANCHORS:'])
    for group in ['train','val','all']:
        s=modes[group];lines.append(f'{group}: all bonds {s["bond_exact_images"]}/{s["images"]}; full graph {s["full_exact_images"]}/{s["images"]} ({s["full_exact_rate"]:.6%}); charge owners {s["charge_correct"]}/{s["charges"]}.')
    lines.extend(['','This diagnostic has an additional coordinate input (atoms[].center). It is NOT a measured performance of the existing box-only detector.',
        'No method reaches 100% full graph correctness here: residual charge-to-atom association remains. All supplied anchor/line bond connections round-trip correctly.',
        'No chemistry canonicalization/SMILES equivalence or OCR expansion is included. GT atom labels are given, including contracted pseudoatom nodes.',
        '', 'Files: report.json (primary), atom_anchor_input_diagnostic.json (additional-anchor mode), samples.csv, failures.csv/json, failure_previews/ and two assembled_graphs*.jsonl.gz files.',
        'Graph JSONL files contain all 14980 assemblies in each input mode. Source code snapshots and manifest/source hashes allow tracing this fixed evaluation.'])
    (run/'report.txt').write_text('\n'.join(lines)+'\n')
    (run/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Full v3 GT assembly evaluation</title><style>body{font:15px sans-serif;margin:24px}pre{white-space:pre-wrap}img{max-width:100%}article{margin:30px 0;border-top:1px solid #ddd}</style><h1>全量 GT 组图评测</h1><pre>'+html.escape('\n'.join(lines))+'</pre><h1>现有 box 中心流程的全部失败样本</h1>'+''.join(cards))
    shutil.copyfile(Path(__file__),run/'source'/Path(__file__).name)
    for name in ['README.md','dataset_config.json','validation_report.json','distribution_report.json']:
        shutil.copyfile(root/name,run/('dataset_'+name))
    with zipfile.ZipFile(run.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for path in sorted(run.rglob('*')):
            if path.is_file():z.write(path,arcname=str(Path(run.name)/path.relative_to(run)))
    with zipfile.ZipFile(run.with_suffix('.zip')) as z:assert z.testzip() is None
    print(json.dumps(dict(primary=report['groups']['all'],additional_anchor_input=modes['all'],anchor_charge_changes=anchor_charge_changes,archive=str(run.with_suffix('.zip').resolve()),bytes=run.with_suffix('.zip').stat().st_size),indent=2),flush=True)

if __name__=='__main__':main()
