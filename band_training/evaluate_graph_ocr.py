"""Exact anonymous-dummy graph and graph+literal-OCR evaluation on frozen outputs."""
import argparse,gzip,json,sys,time,math,hashlib,importlib.metadata,shutil
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import networkx as nx
import torch
import cv2
from PIL import Image,ImageOps,ImageDraw
from .mixed_data import ATOM_CLASSES,letterbox_transform
from .continue_indigo import write
from .audit_graph_references import reference_from_annotation
from .molecular_graph_metrics import property_graph,exact_property_match,normalize_text,mcs_diagnostic


class LiteralOCR:
    def __init__(self,model_dir):
        import easyocr
        self.reader=easyocr.Reader(['en'],gpu=False,detector=False,quantize=False,verbose=False,
            model_storage_directory=str(model_dir.resolve()),download_enabled=False)

    @torch.no_grad()
    def recognize(self,image,box):
        x1,y1,x2,y2=box;candidates=[]
        for margin in [1,3]:
            bounds=(max(0,math.floor(x1-margin)),max(0,math.floor(y1-margin)),min(image.width,math.ceil(x2+margin)),min(image.height,math.ceil(y2+margin)))
            if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:continue
            crop=image.crop(bounds);ratio=64/max(1,crop.height)
            crop=ImageOps.expand(crop.resize((max(1,min(1024,round(crop.width*ratio))),64),Image.Resampling.LANCZOS),border=4,fill='white')
            gray=cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2GRAY);h,w=gray.shape
            rows=self.reader.recognize(gray,horizontal_list=[[0,w,0,h]],free_list=[],decoder='greedy',
                batch_size=1,workers=0,detail=1,paragraph=False,reformat=False)
            text=''.join(str(r[1]) for r in rows);score=float(np.mean([float(r[2]) for r in rows])) if rows else 0.
            candidates.append(dict(text=text,normalized=normalize_text(text),confidence=score,margin=margin,crop_box=list(bounds)))
        best=max(candidates,key=lambda r:r['confidence'] if r['text'] else -1) if candidates else dict(text='',normalized='',confidence=0.)
        return dict(**best,candidates=candidates,selection='highest EasyOCR confidence; no GT text, lexicon or label correction')


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ocr-models',type=Path)
    p.add_argument('--paddle-results',type=Path);p.add_argument('--paddle-policy',choices=['en_single','en_multiscale','latin_multiscale','ensemble_multiscale','en_consensus','latin_consensus','ensemble_consensus'],default='en_consensus')
    p.add_argument('--mcs-examples',type=int,default=10);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    from .dataset_runtime import install_generator_path
    install_generator_path(args.dataset)
    from chem_prep.backend import Indigo
    ig=Indigo()
    if args.paddle_results:
        from .paddle_ocr_replay import PaddleOCRReplay
        ocr=PaddleOCRReplay(args.paddle_results,args.paddle_policy)
    else:
        if args.ocr_models is None:p.error('--ocr-models is required for EasyOCR')
        ocr=LiteralOCR(args.ocr_models)
    started=time.time();counts=Counter();groups=defaultdict(Counter);rows=[];ocr_rows=[];mcs_rows=[]
    with gzip.open(args.cache,'rt',encoding='utf-8') as source,gzip.open(args.output/'graphs.jsonl.gz','wt',encoding='utf-8',compresslevel=3) as graphs:
        for line in source:
            item=json.loads(line);record=item['record'];raw=json.loads((args.dataset/record['annotation']).read_text())
            _,reference_atoms,reference_bonds=reference_from_annotation(ig,raw)
            ref=property_graph(ig,reference_atoms,reference_bonds)
            dummy_refs=[i for i,d in ref.nodes(data=True) if d['token']=='*']
            reftexts={i:reference_atoms[i]['symbol'] for i in dummy_refs}
            ref_labeled=property_graph(ig,reference_atoms,reference_bonds,reftexts)
            atoms=[dict(symbol=ATOM_CLASSES[a['label']] if a['label'] else '*',formal_charge=0) for a in item['atoms']]
            owners=[];valid=not item['unlinked'] and not item.get('untyped_lines',[])
            for q,a in zip(item['charges'],item['charge_assignments']):
                owner=a['atom_index']
                if owner is None:valid=False
                else:
                    atoms[owner]['formal_charge']+={13:-4,14:-2,15:-1,16:1,17:2}[q['label']];owners.append(owner)
            valid=valid and len(set(owners))==len(owners)
            predicted=property_graph(ig,atoms,item['bonds'])
            star_mapping=exact_property_match(predicted,ref) if valid else None
            star=star_mapping is not None
            assert not item['components']['legacy_full_exact'] or star, 'Legacy exact must imply anonymous graph exact: '+item['sample_id']
            pseudo_only_ref=ref.copy()
            for i in dummy_refs:
                if not reference_atoms[i]['is_group']:pseudo_only_ref.nodes[i]['token']=reference_atoms[i]['symbol']
            alias_only=bool(valid and exact_property_match(predicted,pseudo_only_ref) is not None)
            with Image.open(args.dataset/record['image']) as im:image=im.convert('RGB')
            if hasattr(ocr,'set_image'):ocr.set_image(args.dataset/record['image'])
            scale,offset,_=letterbox_transform(image.width,image.height)
            pred_text={};gt_text={};pred_ocr={};gt_ocr={};mapping={int(k):v for k,v in item['atom_mapping'].items()}
            for i,a in enumerate(item['atoms']):
                if a['label']!=0:continue
                x,y,w,h=a['bbox'];box=np.r_[(np.array([x,y])-offset)/scale,(np.array([x+w,y+h])-offset)/scale]
                result=ocr.recognize(image,box);pred_ocr[i]=result;pred_text[i]=result['text']
                j=mapping.get(i);expected=reftexts.get(j)
                correct=expected is not None and result['normalized']==normalize_text(expected)
                counts.update(predicted_dummy_crops=1,matched_dummy_crops=int(expected is not None),matched_dummy_text_correct=int(correct))
                ocr_rows.append(dict(sample_id=item['sample_id'],kind='predicted_crop',predicted_atom=i,reference_atom=j,
                    expected=expected,exact=correct,box=box.tolist(),**result))
            for i in dummy_refs:
                result=ocr.recognize(image,raw['atoms'][i]['bbox']);gt_ocr[i]=result;gt_text[i]=result['text']
                correct=result['normalized']==normalize_text(reftexts[i])
                counts.update(gt_dummy_crops=1,gt_dummy_text_correct=int(correct))
                ocr_rows.append(dict(sample_id=item['sample_id'],kind='gt_crop_diagnostic',reference_atom=i,expected=reftexts[i],exact=correct,
                    box=raw['atoms'][i]['bbox'],**result))
            labeled=property_graph(ig,atoms,item['bonds'],pred_text)
            text_mapping=exact_property_match(labeled,ref_labeled,with_text=True) if star else None
            combined=text_mapping is not None
            oracle_labels=property_graph(ig,reference_atoms,reference_bonds,gt_text)
            oracle_ocr=exact_property_match(oracle_labels,ref_labeled,with_text=True) is not None
            # Exact-text graph success must be a subset of anonymous graph success.
            assert not combined or star
            has_dummy=bool(dummy_refs);has_pseudo=any(reference_atoms[i]['is_group'] for i in dummy_refs)
            for group in ['all','with_dummy' if has_dummy else 'without_dummy']+(['with_alias_or_rsite'] if has_pseudo else []):
                groups[group].update(images=1,graph_star_exact=int(star),graph_ocr_exact=int(combined),gt_boxes_ocr_all_labels_exact=int(oracle_ocr),
                    alias_only_star_exact=int(alias_only))
            c=item['components'];counts.update(images=1,graph_star_exact=int(star),graph_ocr_exact=int(combined),legacy_full_exact=int(c['legacy_full_exact']),
                rejected_unlinked_bands=int(bool(item['unlinked'])),rejected_unassigned_charge=int(not valid and not item['unlinked']))
            difference=''.join('1' if c[k] else '0' for k in ['atom_exact','charge_owner_exact','bond_exact'])
            counts['components_'+difference]+=1
            detail=dict(sample_id=item['sample_id'],image=record['image'],components=c,graph_star_exact=star,graph_ocr_exact=combined,
                alias_only_star_exact=alias_only,gt_boxes_ocr_all_labels_exact=oracle_ocr,has_dummy=has_dummy,
                graph_star_mapping=star_mapping,graph_ocr_mapping=text_mapping,predicted_dummy_ocr=pred_ocr,reference_dummy_labels=reftexts,
                gt_crop_ocr=gt_ocr,valid_assembled_graph=valid,aromatic_normalization_pred=predicted.graph['aromatic_normalization_ok'],
                aromatic_normalization_ref=ref.graph['aromatic_normalization_ok'])
            rows.append(detail)
            graphs.write(json.dumps(dict(sample_id=item['sample_id'],predicted=nx.node_link_data(predicted,edges='edges'),reference=nx.node_link_data(ref,edges='edges'),
                predicted_ocr=nx.node_link_data(labeled,edges='edges'),reference_ocr=nx.node_link_data(ref_labeled,edges='edges')),ensure_ascii=False)+'\n')
            if not star and len(mcs_rows)<args.mcs_examples:
                mcs_rows.append(dict(sample_id=item['sample_id'],**mcs_diagnostic(atoms,item['bonds'],reference_atoms,reference_bonds)))
            if len(rows)%128==0:
                print('SCORED',len(rows),'dummy crops',counts['predicted_dummy_crops'],counts['gt_dummy_crops'],flush=True)
                write(args.output/'progress.json',dict(status='evaluating',images=len(rows)))
    result=dict(status='completed',counts=dict(counts),groups={k:dict(v) for k,v in groups.items()},seconds=time.time()-started,
        ocr=ocr.metadata if args.paddle_results else dict(backend='EasyOCR English g2 pretrained, CPU, recognition only',version=importlib.metadata.version('easyocr'),
            weights_sha256=hashlib.sha256((args.ocr_models/'english_g2.pth').read_bytes()).hexdigest(),uses_gt_labels_for_inference=False,
            normalization='Unicode NFKC + whitespace removal; case, punctuation and R indices preserved',detection_source='predicted class-0 boxes',
            adapted_weights=False),
        metric_scope='Exact property graph isomorphism; existing 13-class vocabulary + anonymous other nodes, formal charge, aromatic-normalized bond type, directed wedge/dash. No partial MCS acceptance; implicit H/isotope/atom-CIP/E-Z are not separately scored.',
        dummy_policy='Model-vocabulary projection: all other nodes are anonymized, including true pseudoatoms and unsupported elements. alias_only_star is also provided, keeping unsupported true element identities.',
        no_ground_truth_geometry_used_for_prediction=True)
    for group,s in result['groups'].items():
        s['graph_star_rate']=s['graph_star_exact']/s['images'];s['graph_ocr_rate']=s['graph_ocr_exact']/s['images']
    for name,obj in [('report.json',result),('samples.json',rows),('ocr_crops.json',ocr_rows),('mcs_diagnostics.json',mcs_rows)]:write(args.output/name,obj)
    snapshots=args.output/'source';snapshots.mkdir()
    for name in ['evaluate_graph_ocr.py','paddle_ocr_replay.py','prepare_paddle_ocr_requests.py','molecular_graph_metrics.py','audit_graph_references.py','cache_final_predictions.py','test_molecular_graph_metrics.py']:
        shutil.copyfile(Path(__file__).with_name(name),snapshots/name)
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
