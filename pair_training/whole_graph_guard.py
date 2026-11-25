"""In-process validation-only whole-molecule retention for all three stages.

Uses the same network boxes, anchors, pair decoding, OCR and E/Z assembly as
the standalone evaluator. Ground truth is loaded only after inference.
"""
from collections import defaultdict
import json
import hashlib
from pathlib import Path
import torch
from PIL import Image
from rdkit import Chem
from torchvision.transforms import functional as TF
from pair_training.balanced_text_training import balanced_validation,retention_eligible


class WholeGraphGuard:
    def __init__(self,model,anchor,ck,ocr,dataset,device,limit=0,frozen_graphs=False):
        if limit and limit<3:raise ValueError('Retention smoke must include all three scales')
        from scripts.evaluate_structure_backends import infer_graph
        self.model,self.anchor,self.ck,self.ocr=model,anchor,ck,ocr
        self.device=device;self.infer_graph=infer_graph;self.frozen_graphs=frozen_graphs
        self.graph_cache={};self.label_cache={};self.rows=[]
        root=Path(__file__).resolve().parents[1]
        self.classes=root/'dataset_pair_real_20261015/annotations/classes.json'
        from scripts.build_text_benchmark import FRAGMENTS
        self.known=set(FRAGMENTS)
        for domain,source in [('old',root/'dataset_text_training_20261019'),('new',dataset)]:
            rows=[r for r in json.loads((source/'manifest.json').read_text()) if r['split']=='val']
            if limit:
                # Within each scale interleave controls, known aliases and
                # free literals, then interleave scales for the smoke subset.
                groups=defaultdict(lambda:defaultdict(list))
                for r in rows:groups[r['native_size']][self.label_type(r)].append(r)
                rows=[]
                for size,types in sorted(groups.items()):
                    while any(types.values()):
                        for kind in ['control','known','literal']:
                            if types[kind]:rows.append(types[kind].pop(0))
            rows=balanced_validation(rows,limit,lambda r:r['native_size'])
            hashes=json.loads((source/'truth_hashes.json').read_text())
            for r in rows:
                for key in ['image','truth']:
                    assert hashlib.sha256((source/r[key]).read_bytes()).hexdigest()==hashes[r[key]],r[key]
            self.rows.extend((domain,source,r) for r in rows)
        self.baseline=None;self.last_cases=[]

    def label_type(self,row):
        if not row['labels']:return 'control'
        return 'known' if all(l['text'] in self.known for l in row['labels']) else 'literal'

    @torch.no_grad()
    def evaluate(self):
        from pair_training.model import predicted_atoms
        from pair_training.text_atoms import apply_text_anchors
        from pair_training.chemistry import build_rdkit_molecule
        from pair_training.notebook_ocr import recognize_multiscale,labeled_cxsmiles,expand_labels
        from pair_training.label_metrics import label_graph_equal
        self.model.eval();self.anchor.eval();self.ocr.eval();counts=defaultdict(lambda:dict(correct=0,total=0));cases=[]
        for domain,source,r in self.rows:
            key=str(source/r['image']);image=Image.open(key).convert('RGB')
            if self.frozen_graphs and key in self.graph_cache:
                graph,mol,chem=self.graph_cache[key]
            else:
                x=TF.normalize(TF.to_tensor(TF.resize(image,[640,640])),[.485,.456,.406],[.229,.224,.225])[None].to(self.device)
                features,out=self.model.extract(x);atoms=predicted_atoms(out,0,cutoff=self.ck['args']['atom_cutoff'])
                atoms['raster_resolution']=min(image.size)
                atoms,_=apply_text_anchors(atoms,image,features,self.anchor)
                graph=self.infer_graph(atoms,image,features,self.model,self.ck,self.classes)
                mol,chem=build_rdkit_molecule(graph,infer_double_stereo=True)
                if self.frozen_graphs:self.graph_cache[key]=(graph,mol,chem)
            # Pair-only updates keep the entity/anchor/OCR unchanged. Cache by
            # the exact OCR inputs; disable this cache when OCR itself trains.
            cachekey=(key,json.dumps(graph['atoms'],sort_keys=True))
            if not self.frozen_graphs and cachekey in self.label_cache:labels=self.label_cache[cachekey]
            else:
                labels,_=recognize_multiscale(graph,image,scales=(1,2,3,4),margins=(2,4),min_confidence=.5,recognizer=self.ocr.recognize_crop)
                if not self.frozen_graphs:self.label_cache[cachekey]=labels
            cx=None;smiles=None
            if chem['valid']:
                _,c=labeled_cxsmiles(mol,graph,labels);_,expanded=expand_labels(mol,graph,labels)
                cx=c.get('cxsmiles');smiles=expanded.get('smiles')
            truth=json.loads((source/r['truth']).read_text())
            if truth.get('expanded_truth_available',True):correct=smiles==truth['expected_smiles']
            else:correct=label_graph_equal(Chem.MolFromSmiles(cx) if cx else None,Chem.MolFromSmiles(truth['expected_cxsmiles']))
            kind=self.label_type(r)
            cases.append(dict(domain=domain,image=r['image'],native_size=r['native_size'],label_type=kind,
                              correct=bool(correct),valid=chem['valid'],predicted_labels=labels,
                              truth_labels=r['labels'],smiles=smiles,cxsmiles=cx,graph=graph))
            for group in [domain,domain+':'+str(r['native_size']),domain+':'+kind,
                          domain+':'+str(r['native_size'])+':'+kind]:
                counts[group]['total']+=1;counts[group]['correct']+=bool(correct)
        result=dict(counts)
        self.last_cases=cases
        if self.baseline is None:self.baseline=result
        return result

    def eligible(self,metrics):
        return retention_eligible(self.baseline,metrics)


def load_frozen_guard(bundle,ocr,dataset,device,limit):
    from pair_training.train import load_model
    from pair_training.model import state_hash
    from pair_training.text_sequence import NativeROITextHead
    ck=torch.load(bundle/'pair.pth',map_location='cpu',weights_only=False)
    model=load_model(bundle/'base.pth',device,ck['architecture']).eval().requires_grad_(False)
    assert state_hash(model.base)==ck['base_state_sha256']
    model.pair_head.load_state_dict(ck['pair_head'])
    ac=torch.load(bundle/'anchor.pth',map_location='cpu',weights_only=False)
    assert ac['base_hash']==ck['base_state_sha256']
    anchor=NativeROITextHead(**ac['config']).to(device).eval().requires_grad_(False)
    anchor.load_state_dict(ac['state_dict'])
    return WholeGraphGuard(model,anchor,ck,ocr,dataset,device,limit,frozen_graphs=True)
