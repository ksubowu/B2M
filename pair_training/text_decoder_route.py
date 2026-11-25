"""Shared encoder, frozen original decoder, and a trainable text decoder.

Routing inspects predictions and chemistry only, never ground truth or words.
"""
from collections import Counter
import copy
import torch
from pair_training.chemistry import build_rdkit_molecule
from pair_training.model import state_hash

POLICY=dict(version='shared_encoder_text_decoder_guard_v1',atom_cutoff=.4,
            plain_reference=True,invalid_or_fragmented_fallback=True)


def graph_quality(graph,image):
    from pair_training.notebook_ocr import group_text_targets
    parent={a['index']:a['index'] for a in graph['atoms']};degree=Counter()
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for b in graph['bonds']:
        i,j=b['begin_atom_idx'],b['end_atom_idx'];parent[find(i)]=find(j);degree[i]+=1;degree[j]+=1
    blocked=set()
    if len({find(i) for i in parent})>1:
        for target in group_text_targets(graph,image):
            blocked.update(i for i in target['merged_atom_indices'] if i!=target['index'] and degree[i]==0)
    components=len({find(i) for i in parent if i not in blocked})
    _,chem=build_rdkit_molecule(graph,infer_double_stereo=True)
    return dict(valid=chem['valid'],connected=components==1,
                has_text=any(a['symbol']=='*' for a in graph['atoms']))


def choose_route(candidate,reference):
    if reference['valid'] and reference['connected']:
        if not reference['has_text']:return 'plain_reference'
        if not candidate['valid'] or not candidate['connected']:return 'invalid_or_fragmented_fallback'
    return 'learner'


def enable_reference_heads(model,decoder,anchor):
    model.original_box_decoder=decoder.eval().requires_grad_(False)
    model.original_anchor=anchor.eval().requires_grad_(False)
    model.text_entity_route=dict(POLICY)


def reference_payload(model):
    return dict(text_entity_route=POLICY,original_box_decoder=model.original_box_decoder.state_dict(),
                original_anchor=dict(config=model.original_anchor.config,state_dict=model.original_anchor.state_dict()),
                original_box_decoder_hash=state_hash(model.original_box_decoder),
                original_anchor_hash=state_hash(model.original_anchor),
                shared_feature_hashes={k:state_hash(getattr(model.base,k)) for k in ['backbone','encoder']})


def load_reference_heads(model,payload,device):
    from pair_training.text_sequence import NativeROITextHead
    if payload['text_entity_route']!=POLICY:raise ValueError('Unknown text decoder routing policy')
    for name,expected in payload['shared_feature_hashes'].items():
        assert state_hash(getattr(model.base,name))==expected,'Original decoder requires the same frozen image features'
    decoder=copy.deepcopy(model.base.box_decoder)
    decoder.load_state_dict(payload['original_box_decoder'],strict=True)
    ac=payload['original_anchor'];anchor=NativeROITextHead(**ac['config']).to(device)
    anchor.load_state_dict(ac['state_dict'],strict=True)
    assert state_hash(decoder)==payload['original_box_decoder_hash']
    assert state_hash(anchor)==payload['original_anchor_hash']
    enable_reference_heads(model,decoder,anchor)
