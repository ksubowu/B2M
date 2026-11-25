"""Package the unique-endpoint correspondence comparison and illustrative cases."""
import argparse,json,zipfile,re,html,shutil,gzip
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .mixed_data import prepare_annotation,letterbox_transform
from .endpoint_assignment import associate_unique_endpoints

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
    r=json.loads((run/'report.json').read_text());assert r['status']=='completed' and r['forward_baseline_reproduced']
    root=Path(r['dataset']);failures=json.loads((run/'failures.json').read_text());samples=json.loads((run/'samples.json').read_text());by={s['sample_id']:s for s in samples}
    primary=[f for f in failures if f['method']=='exclusive_box'];assert len(primary)==29 and not any(f['failed_bonds'] for f in primary)
    (run/'exclusive_box_remaining_charge_failures.json').write_text(json.dumps(primary,indent=2,ensure_ascii=False))
    examples=[('isolated_atom',next(f for f in failures if f['method']=='reverse_nearest' and by[f['sample_id']]['isolated_atoms']>0)),
              ('long_label',next(f for f in failures if f['method']=='reverse_nearest' and not by[f['sample_id']]['isolated_atoms'])),
              ('global_center_ambiguity',next(f for f in failures if f['method']=='global_center' and f['failed_bonds']))]
    folder=run/'examples';folder.mkdir(exist_ok=True);cards=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    details=[]
    for name,f in examples:
        source=json.loads((root/f['annotation']).read_text());bt=json.loads((root/source['band_target']).read_text());lt=json.loads((root/source['line_target']).read_text())
        ann,_,labels,quads=prepare_annotation(source,bt);scale,offset,shape=letterbox_transform(source['width'],source['height'])
        lines=np.asarray(lt['lines'])*scale+offset;boxes=np.asarray([a['bbox'] for a in ann['atoms']]);centers=boxes[:,:2]+boxes[:,2:]/2
        _,old=associate_unique_endpoints(boxes,lines,'reverse_nearest');_,new=associate_unique_endpoints(boxes,lines,'exclusive_box')
        points=np.asarray(old['unique_endpoint_coordinates']);mapping_old=np.asarray(old['box_to_endpoint']);mapping_new=np.asarray(new['box_to_endpoint'])
        changed=np.flatnonzero(mapping_old!=mapping_new);im=Image.open(root/f['image']).convert('RGB');base=Image.new('RGB',(640,640),'white');base.paste(im.resize(shape,Image.Resampling.LANCZOS),tuple(offset))
        left=base.copy();right=base.copy()
        for panel,mapping,color in [(left,mapping_old,'red'),(right,mapping_new,'green')]:
            d=ImageDraw.Draw(panel)
            for x,y in points:d.ellipse([x-1,y-1,x+1,y+1],fill='#0090d0')
            for i in changed:
                x,y,w,h=boxes[i];d.rectangle([x,y,x+w,y+h],outline=color,width=2)
                d.text((x,max(0,y-15)),f'{i}:{ann["atoms"][i]["symbol"]}',font=font,fill=color,stroke_width=1,stroke_fill='white')
                if mapping[i]>=0:d.line([tuple(centers[i]),tuple(points[mapping[i]])],fill=color,width=2)
                else:d.text(tuple(centers[i]),'no line endpoint',font=font,fill=color)
        canvas=Image.new('RGB',(1920,720),'white');draw=ImageDraw.Draw(canvas)
        for col,(panel,title) in enumerate(zip([base,left,right],['Original image','Box -> nearest unique endpoint','Exclusive box-aware correspondence'])):
            canvas.paste(panel,(640*col,40));draw.text((640*col+5,8),title,font=font,fill='black')
        draw.text((5,688),name+' | '+f['sample_id'],font=font,fill='black');filename='examples/'+name+'.png';canvas.save(run/filename)
        details.append(dict(example=name,sample_id=f['sample_id'],old=old,new=new,changed_boxes=changed.tolist()))
        cards.append(f'<h2>{html.escape(name)}: {html.escape(f["sample_id"])}</h2><a href="{filename}"><img style="max-width:100%" src="{filename}"></a>')
    (run/'example_assignments.json').write_text(json.dumps(details,indent=2,ensure_ascii=False))
    t=r['topology'];text=['Unique line-endpoint -> atom-box correspondence: full GT evaluation','',
        f'Dataset: {root}',f'Unique images: {t["images"]}; atom boxes: {t["atoms"]}; unique line endpoints: {t["unique_endpoints"]}.',
        f'Isolated atoms: {t["isolated_atoms"]}, in {t["images_with_isolated_atoms"]} images. Unique endpoint count equals incident-atom count in ALL images.',
        f'Pure reverse nearest neighbor has collisions in {t["raw_nn_collision_images"]} images, involving {t["raw_nn_collision_endpoints"]} endpoint nodes; {t["raw_nn_uncovered_endpoints"]} endpoint nodes receive no box.',
        '', 'Fixed-method comparisons (all 14980 images):']
    for m in ['forward','reverse_nearest','global_center','exclusive_box']:
        s=r['methods'][m]['all'];text.append(f'{m}: all bonds exact {s["bond_exact"]}/{s["images"]} ({s["bond_exact_rate"]:.6%}); typed bonds {s["bond_tp"]}/{s["bonds"]}; full graph including charge {s["full_exact"]}/{s["images"]} ({s["full_exact_rate"]:.6%}).')
    text.extend(['', 'reverse_nearest collision handling: ambiguous endpoint ownership is left unresolved; no arbitrary overwrite or GT-based choice.',
        'global_center: one-to-one Euclidean Hungarian matching; extra boxes may remain isolated.',
        'exclusive_box: lock mutually unique center matches within 0.01px (rounding allowance), then one-to-one Hungarian matching using point-to-box-boundary distance + 0.02 center distance. This uses box geometry and line endpoints only.',
        'No annotated atom.center, source/target IDs, atom classes, GT degrees, or chemical owner fields enter endpoint correspondence. Band provides edge class. Charge association remains joint_valence.',
        'The prior forward baseline and annotation/target stream hashes are reproduced exactly.',
        '', 'exclusive_box results by split:'])
    for group in ['train','val']:
        s=r['methods']['exclusive_box'][group];text.append(f'{group}: bonds exact {s["bond_exact"]}/{s["images"]}; full graph {s["full_exact"]}/{s["images"]}.')
    text.extend(['', 'Best coordinate-only method recovers all 368318 typed/directed bonds on this GT corpus. The remaining 29 images contain 30 incorrectly associated charge instances; full graph accuracy is 99.806409%.',
        'These are EXACT ground-truth endpoints. No claim is made that independently predicted float endpoints are identical: predicted outputs need separately validated clustering/tolerance handling.',
        'This is geometry postprocessing validation, not neural end-to-end/OCR/expanded-SMILES accuracy.',
        '', 'API: assemble_band_lines(atom_boxes_xywh, bond_instances, line_association="exclusive_box")',
        'CLI: python -m band_training.assemble_band_lines --input GEOMETRY.json --output GRAPH.json --line-association exclusive_box',
        'Each paired bond instance has label (0..5), quad (4x2), line (2x2); all coordinates must share the same pixel system.',
        'The previous forward behavior remains the default, preserving prior evaluations. The new method is available explicitly via the API/CLI.',
        '', 'Files: report.json, samples.csv, failures.json (all compared methods), exclusive_box_remaining_charge_failures.json, assembled_graphs.jsonl.gz (all four methods for all images), examples/, source/.'])
    (run/'report.txt').write_text('\n'.join(text)+'\n')
    (run/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Unique endpoint correspondence evaluation</title><style>body{font:15px sans-serif;margin:24px}pre{white-space:pre-wrap}</style><h1>唯一 line 端点与 atom box 的对应评测</h1><pre>'+html.escape('\n'.join(text))+'</pre>'+''.join(cards))
    for name in ['assemble_band_lines.py','test_endpoint_assignment.py','finalize_endpoint_assignment.py']:
        shutil.copyfile(Path(__file__).with_name(name),run/'source'/name)
    with gzip.open(run/'assembled_graphs.jsonl.gz','rt',encoding='utf-8') as f:assert sum(1 for _ in f)==14980
    with zipfile.ZipFile(run.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for file in sorted(run.rglob('*')):
            if file.is_file():z.write(file,arcname=str(Path(run.name)/file.relative_to(run)))
    with zipfile.ZipFile(run.with_suffix('.zip')) as z:assert z.testzip() is None
    print('ZIP',run.with_suffix('.zip').resolve(),'bytes',run.with_suffix('.zip').stat().st_size,flush=True)

if __name__=='__main__':main()
