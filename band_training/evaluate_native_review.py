"""Apply an unchanged geometric assembler to native-ownership review ZIP data."""
import argparse,csv,hashlib,io,json,re,zipfile,html,shutil
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .mixed_data import prepare_annotation,letterbox_transform
from .geometry import link_many,edge_key,target_edges,aggregate,assemble_band_lines
from .train import charge_exact
from .charge_association import assign_charges


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--zip',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--overwrite',action='store_true')
    p.add_argument('--band-lines',action='store_true',help='Use paired line endpoints for connectivity, band for class/region.')
    p.add_argument('--charge-association',choices=['legacy','joint_valence','safe_adaptive'],default='legacy')
    args=p.parse_args();primary='band_line' if args.band_lines else 'band_directional'
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=args.overwrite)
    for name in ['graphs','failures','geometry_inputs']:(args.output/name).mkdir(exist_ok=args.overwrite)
    methods=['kdtree','band_directional']+(['band_line'] if args.band_lines else [])
    rows={m:[] for m in methods};cases=[];failures=[];paired=defaultdict(list)
    with zipfile.ZipFile(args.zip) as z:
        assert z.testzip() is None
        names=[n for n in z.namelist() if n.endswith('/data_manifest.json')];assert len(names)==1
        prefix=names[0].rsplit('/',1)[0]+'/'
        records=json.loads(z.read(names[0]));assert len(records)==112
        assert len({r['image_id'] for r in records})==56
        for record in records:
            source=json.loads(z.read(prefix+record['annotation']));target=json.loads(z.read(prefix+record['band_target']))
            assert source['ownership']['geometry_verified']
            image=Image.open(io.BytesIO(z.read(prefix+record['image']))).convert('RGB')
            assert image.size==(source['width'],source['height'])
            assert np.allclose(target['quads'],[b['quad'] for b in source['bands']],atol=.001)
            ann,_,labels,quads=prepare_annotation(source,target)
            assert len(quads)==len(ann['bonds'])
            lines=None;instances=None
            if args.band_lines:
                lt=json.loads(z.read(prefix+record['line_target']))
                assert lt['labels']==labels and lt['size']==target['size']
                assert len(lt['lines'])==len(quads)
                assert np.allclose(lt['lines'],[v['endpoints'] for v in source['lines']],atol=.001)
                scale,offset,_=letterbox_transform(source['width'],source['height'])
                lines=(np.asarray(lt['lines'],float).reshape(-1,2,2)*scale+offset).tolist()
                # One bond instance owns both outputs. Only coordinate arrays
                # and band classes enter assembly; source/target/bond IDs do not.
                instances=[dict(label=l,quad=q,line=line) for l,q,line in zip(labels,quads,lines)]
            boxes=[a['bbox'] for a in ann['atoms']];truth=target_edges(ann);result={};pairs_by_method={}
            for method in methods:
                if method=='band_line':
                    assembled=assemble_band_lines(boxes,instances)
                    pairs=[None]*len(instances)
                    for edge in assembled['bonds']:pairs[edge['bond_instance']]=(edge['begin'],edge['end'])
                else:pairs=link_many(quads,boxes,method=method)
                pairs_by_method[method]=pairs
                predicted=Counter(edge_key(*pair,label) for pair,label in zip(pairs,labels) if pair is not None)
                tp=sum((truth&predicted).values());unlinked=sum(pair is None for pair in pairs)
                result[method]=dict(tp=tp,fp=len(quads)-tp,fn=len(quads)-tp,unlinked=unlinked,
                                    exact=int(predicted==truth and not unlinked),gt=len(quads))
                rows[method].append(result[method])
            atoms=[dict(label=a['category_id'],bbox=a['bbox']) for a in ann['atoms']]
            charges=[dict(label=q['category_id'],bbox=q['bbox']) for q in ann['charges']]
            legacy_charge_ok=charge_exact(atoms+charges,atoms,{i:i for i in range(len(atoms))},ann)
            pairs=pairs_by_method[primary]
            charge_bonds=[dict(begin=pair[0],end=pair[1],label=label) for pair,label in zip(pairs,labels) if pair is not None]
            charge_links=assign_charges(atoms,charges,charge_bonds,method=args.charge_association,lines=lines)
            charge_ok=charge_exact(atoms+charges,atoms,{i:i for i in range(len(atoms))},ann,assignments=charge_links)
            assigned=[];charge_details=[];formal=[0]*len(atoms)
            for qi,charge in enumerate(charges):
                x,y,w,h=charge['bbox'];center=np.array([x+w/2,y+h/2]);scores=[]
                for i,(ax,ay,aw,ah) in enumerate(boxes):
                    distance=np.linalg.norm(np.maximum(np.maximum([ax,ay]-center,center-[ax+aw,ay+ah]),0))
                    scores.append(distance+.02*np.linalg.norm(center-[ax+aw/2,ay+ah/2]))
                i=charge_links[qi]['atom_index'];value={13:-4,14:-2,15:-1,16:1,17:2}[charge['label']]
                if i is not None:formal[i]+=value
                assigned.append(dict(atom_index=i,value=value,bbox=charge['bbox']))
                charge_details.append(dict(charge_index=qi,value=value,predicted_atom=i,expected_atom=ann['charges'][qi]['atom_index']))
            pairs=pairs_by_method[primary];errors=[];geometry_edges=[]
            for i,(pair,bond,label,quad) in enumerate(zip(pairs,ann['bonds'],labels,quads)):
                expected=(bond['begin'],bond['end']);okay=pair is not None and edge_key(*pair,label)==edge_key(*expected,label)
                geometry_edges.append(dict(band_index=i,ends=pair,category_id=label+1))
                if not okay:
                    q=np.asarray(quad);ends=np.stack((q[:2].mean(0),q[2:].mean(0)))
                    errors.append(dict(band_index=i,category_id=label+1,predicted=pair,expected=expected,
                        endface_distance=float(np.linalg.norm(ends[1]-ends[0])),
                        direction_only=bool(pair is not None and sorted(pair)==sorted(expected)),quad=quad))
            status=dict(sample_id=record['sample_id'],image_id=record['image_id'],variant=record['variant'],
                        atoms=len(atoms),bonds=len(quads),charges=len(charges),charge_exact=bool(charge_ok),
                        full_graph_exact=bool(result[primary]['exact'] and charge_ok),
                        methods=result,errors=errors,charge_assignments=charge_details,legacy_charge_exact=bool(legacy_charge_ok),charge_assignment_audit=charge_links)
            cases.append(status);paired[record['image_id']].append(status)
            stem=re.sub('[^a-zA-Z0-9_.-]+','_',record['sample_id'])
            graph=dict(sample_id=record['sample_id'],association=primary,coordinate_system='640x640 letterbox',
                atoms=[dict(index=i,symbol=source['atoms'][i]['symbol'],bbox=bbox,formal_charge=formal[i]) for i,bbox in enumerate(boxes)],
                charges=assigned,bands=[dict(quad=q,category_id=l+1) for q,l in zip(quads,labels)],
                bonds=[b for b in geometry_edges if b['ends'] is not None],
                unlinked_band_indices=[b['band_index'] for b in geometry_edges if b['ends'] is None])
            if lines is not None:
                graph['lines']=lines
                (args.output/'geometry_inputs'/f'{stem}.json').write_text(json.dumps(dict(atom_boxes=boxes,bond_instances=instances),indent=2))
            (args.output/'graphs'/f'{stem}.json').write_text(json.dumps(graph,indent=2))
            if errors or not charge_ok:
                failures.append(status)
                scale,offset,shape=letterbox_transform(*image.size);base=Image.new('RGB',(640,640),'white');base.paste(image.resize(shape,Image.Resampling.LANCZOS),tuple(offset))
                gt=base.copy();pr=base.copy();gd=ImageDraw.Draw(gt);pd=ImageDraw.Draw(pr)
                for i,(x,y,w,h) in enumerate(boxes):
                    for draw in [gd,pd]:draw.rectangle([x,y,x+w,y+h],outline='#1674c0');draw.text((x,max(0,y-10)),str(i),fill='#1674c0')
                bad={e['band_index'] for e in errors}
                for i,(q,bond,pair) in enumerate(zip(quads,ann['bonds'],pairs)):
                    poly=[tuple(v) for v in q];gd.line(poly+[poly[0]],fill='red' if i in bad else '#15924d',width=1)
                    if i in bad:
                        centers=np.asarray(boxes)[:,:2]+np.asarray(boxes)[:,2:]/2
                        gd.line([tuple(centers[bond['begin']]),tuple(centers[bond['end']])],fill='blue',width=2)
                        if pair:pd.line([tuple(centers[pair[0]]),tuple(centers[pair[1]])],fill='red',width=2)
                        pd.text(tuple(np.asarray(q).mean(0)),f'B{i}',fill='red')
                for qi,(charge,detail) in enumerate(zip(charges,charge_details)):
                    x,y,w,h=charge['bbox'];qc=(x+w/2,y+h/2)
                    centers=np.asarray(boxes)[:,:2]+np.asarray(boxes)[:,2:]/2
                    for draw,ai in [(gd,detail['expected_atom']),(pd,detail['predicted_atom'])]:
                        draw.rectangle([x,y,x+w,y+h],outline='#d000c0',width=2)
                        if ai is not None:draw.line([qc,tuple(centers[ai])],fill='#d000c0',width=2)
                        draw.text((x,y+3),f'Q{qi}:{detail["value"]:+d}->{ai}',fill='#d000c0')
                canvas=Image.new('RGB',(1920,700),'white');font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);draw=ImageDraw.Draw(canvas)
                for col,(im,title) in enumerate(zip([base,gt,pr],['Updated image','GT bands; blue = expected failed links','Red = incorrect assembled links'])):
                    canvas.paste(im,(col*640,35));draw.text((col*640+5,5),title,font=font,fill='black')
                draw.text((5,680),record['sample_id'],font=font,fill='black');canvas.save(args.output/'failures'/f'{stem}.png')
    variant={}
    for name in sorted({c['variant'] for c in cases}):
        group=[c for c in cases if c['variant']==name]
        variant[name]={m:aggregate([c['methods'][m] for c in group]) for m in methods}
        variant[name]['full_graph_exact']=sum(c['full_graph_exact'] for c in group)/len(group)
    paired_rows=[]
    for key,group in paired.items():
        assert len(group)==2
        paired_rows.append(dict(image_id=key,both_variants_bonds_exact=all(c['methods'][primary]['exact'] for c in group),
            both_variants_full_exact=all(c['full_graph_exact'] for c in group),
            bw_exact=next(c['methods'][primary]['exact'] for c in group if c['variant']=='bw'),
            color_exact=next(c['methods'][primary]['exact'] for c in group if c['variant']=='color')))
    report=dict(input_zip=str(args.zip.resolve()),input_sha256=hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        method=primary,geometry_sha256=hashlib.sha256(Path(__file__).with_name('geometry.py').read_bytes()).hexdigest(),
        assembly_inputs='GT boxes and paired band/line coordinates' if args.band_lines else 'GT boxes and ordered band quads',
        no_gt_ids_used_for_association=True,
        source_cases=len(paired),rendered_images=len(cases),variants=variant,
        methods={m:aggregate(rows[m]) for m in methods},
        full_graph_exact_images=sum(c['full_graph_exact'] for c in cases),
        full_graph_exact=sum(c['full_graph_exact'] for c in cases)/len(cases),
        paired_bond_exact=sum(r['both_variants_bonds_exact'] for r in paired_rows),
        paired_full_exact=sum(r['both_variants_full_exact'] for r in paired_rows),
        failures=len(failures),failed_bands=sum(len(c['errors']) for c in cases),charge_exact_images=sum(c['charge_exact'] for c in cases),
        charge_association=args.charge_association,legacy_charge_exact_images=sum(c['legacy_charge_exact'] for c in cases),
        failure_reasons=dict(short_band_rejected=sum(e['endface_distance']<3 and e['predicted'] is None for c in cases for e in c['errors']),
            wrong_atom_links=sum(e['predicted'] is not None for c in cases for e in c['errors']),
            direction_only=sum(e['direction_only'] for c in cases for e in c['errors'])))
    for name,data in [('report.json',report),('samples.json',cases),('failures.json',failures),('paired_cases.json',paired_rows)]:
        (args.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
    with (args.output/'paired_cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired_rows[0]));writer.writeheader();writer.writerows(paired_rows)
    with (args.output/'samples.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['sample_id','image_id','variant','atoms','bonds','charges','bonds_exact','charge_exact','full_graph_exact','failed_bands']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c in cases:writer.writerow({k:v for k,v in dict(**{k:c[k] for k in fields if k in c},
            bonds_exact=bool(c['methods'][primary]['exact']),failed_bands=','.join(str(e['band_index']) for e in c['errors'])).items()})
    summary=['Updated native ownership review: fixed postprocessing evaluation','',
        'Input contains 56 source cases, each rendered in bw and color (112 images). These are new layouts, not edited coordinates on the former images.',
        'Input to assembly: ground-truth atom/charge boxes, classes and paired band/line geometry. No neural detection or OCR.' if args.band_lines else 'Input to assembly: ground-truth boxes, classes and bands.',
        f'Association method {primary}. GT source/target identities and native ownership IDs are used only after assembly for scoring.',
        'All-bonds exact requires matching every connection, type and wedge/dash direction, with no missing/extra/unlinked bands.',
        'Full graph exact additionally requires correct charge-to-atom association. Atom nodes/classes are given inputs.',
        'This is a geometry assembly evaluation, not a SMILES canonicalization or end-to-end recognition result.','']
    for name,v in variant.items():
        m=v[primary];summary.append(f'{name}: all bonds exact {m["exact"]}/{m["images"]} ({m["graph_exact"]:.4%}); full graph exact {v["full_graph_exact"]:.4%}; bond F1 {m["f1"]:.4%}.')
    m=report['methods'][primary];summary.extend([f'Combined: all bonds exact {m["exact"]}/112 ({m["graph_exact"]:.4%}); full graph exact {report["full_graph_exact_images"]}/112 ({report["full_graph_exact"]:.4%}); bond F1 {m["f1"]:.4%}.',
        f'Both renderings correct per source: bonds {report["paired_bond_exact"]}/56; full graph {report["paired_full_exact"]}/56.',
        f'Failed bands: {report["failed_bands"]}; reasons: {report["failure_reasons"]}.',
        f'Charge association method: {args.charge_association}; charge failure images: {len(cases)-report["charge_exact_images"]}.',
        '', 'graphs/: assembled graph for every rendering. samples.csv: per-rendering scores. paired_cases.csv: source-level paired scores.',
        'failures.json and failures/: detailed incorrect links and visualizations (magenta charge links, red wrong bond links).'])
    (args.output/'report.txt').write_text('\n'.join(summary)+'\n')
    if args.band_lines:
        with (args.output/'report.txt').open('a') as f:
            f.write('\nBand/line pairing: one supplied instance, paired by input array position, no source/target/bond IDs consumed.\n')
            f.write('Band supplies class/region; line supplies ordered endpoints. KD-tree built from BOX centers, not annotated atoms[].center. No 3-pixel band rejection applies.\n')
            f.write(f'Charge association: {args.charge_association}. This evaluates annotation geometry, not learned line predictions; the prior trained model has no new line head.\n')
    cards=[]
    for c in failures:
        stem=re.sub('[^a-zA-Z0-9_.-]+','_',c['sample_id']);path='failures/'+stem+'.png'
        cards.append(f'<h2>{html.escape(c["sample_id"])}</h2><p>Bad bands: {[e["band_index"] for e in c["errors"]]}; charge correct: {c["charge_exact"]}</p><a href="{path}"><img loading="lazy" src="{path}" style="max-width:100%"></a>')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Native review assembly failures</title><h1>Updated 56-case assembly evaluation</h1><pre>'+html.escape('\n'.join(summary))+'</pre>'+''.join(cards))
    source=args.output/'source';source.mkdir(exist_ok=True)
    for filename in ['geometry.py','evaluate_native_review.py','mixed_data.py','assemble_band_lines.py','charge_association.py']:
        shutil.copyfile(Path(__file__).with_name(filename),source/filename)
    archive=args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():output.write(path,arcname=str(Path(args.output.name)/path.relative_to(args.output)))
    with zipfile.ZipFile(archive) as output:assert output.testzip() is None
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
