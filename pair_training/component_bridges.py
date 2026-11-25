"""Opt-in recovery of disconnected bonds with strict native ink evidence.

Existing bonds are never changed. No ground truth, OCR word dictionary or
uniform lowering of model thresholds. Connected graphs return immediately.
"""
import numpy as np
import cv2

POLICY=dict(version='native_component_bridges_v1',minimum_continuity=.90,
            ink_threshold=.40,minimum_length_ratio=.60,maximum_length_ratio=1.40,
            minimum_conditional_type=.60,allowed_types=[1,2,3])
VALENCE={'C':4,'N':3,'O':2,'F':1,'Cl':1,'Br':1,'I':1,'P':5,'S':6,'H':1,'B':3,'Si':4,'*':1}


def native_line_evidence(image,a,b,category):
    """Constant-offset parallel strokes; a crossing/angled line cannot zigzag."""
    gray=np.asarray(image.convert('RGB')).min(-1).astype(np.float32)/255.
    ink=1-gray;a=np.asarray(a,float);b=np.asarray(b,float);delta=b-a;length=float(np.linalg.norm(delta))
    if length<3:return None
    normal=np.array([-delta[1],delta[0]])/length
    offsets=np.arange(-max(2.,.20*length),max(2.,.20*length)+.25,.5)
    t=np.linspace(.22,.78,max(24,int(length*2)))
    points=a[None,None,:]+t[None,:,None]*delta[None,None,:]+offsets[:,None,None]*normal[None,None,:]
    sample=cv2.remap(ink,points[:,:,0].astype(np.float32),points[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    continuity=(sample>=POLICY['ink_threshold']).mean(1)
    good=continuity>=POLICY['minimum_continuity'];groups=[]
    for i in np.flatnonzero(good):
        if groups and i==groups[-1][-1]+1:groups[-1].append(int(i))
        else:groups.append([int(i)])
    if len(groups)!=category:return None
    centers=[float(np.average(offsets[g],weights=sample[g].mean(1)+1e-6)) for g in groups]
    widths=[float(len(g)*.5) for g in groups]
    # A broad filled wedge is not accepted as a single line.
    if any(w>max(3.,.14*length) for w in widths):return None
    tolerance=max(1.2,.08*length)
    if category==1 and abs(centers[0])>tolerance:return None
    if category==2:
        gap=centers[1]-centers[0]
        if not 1.2<=gap<=.24*length:return None
        if min(abs(np.mean(centers)),min(map(abs,centers)))>tolerance:return None
    if category==3:
        if min(np.diff(centers))<1.2 or abs(centers[1])>tolerance:return None
    return dict(continuity=float(min(float(continuity[g].max()) for g in groups)),
                parallel_strokes=category,offsets=centers,widths=widths,length=length)


def bridge_candidates(graph,image,boxes,centers,pairs,probabilities,score_mode='max_type'):
    """Return additional decode_reference-style bonds and an inspectable audit."""
    atoms=graph['atoms'];n=len(atoms);audit=dict(policy=POLICY,added=[],components_before=0)
    if n<2:return [],audit
    parent=list(range(n));degree=np.zeros(n,int);used=np.zeros(n,float)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    orders={0:1,1:1,2:2,3:3,4:1.5,5:1,6:1}
    lengths=[]
    # Assembly sorts score ties spatially; recover exact raw indices by boxes
    # instead of assuming NMS and graph indices have the same order.
    raw_to_graph={};graph_to_raw={}
    for node in atoms:
        matches=[i for i,b in enumerate(boxes) if np.allclose(b,node['box'],atol=1e-5,rtol=0)]
        if len(matches)!=1:
            audit['skipped']='ambiguous_atom_mapping';return [],audit
        raw_to_graph[matches[0]]=node['index'];graph_to_raw[node['index']]=matches[0]
    for bond in graph['bonds']:
        i,j=bond['begin_atom_idx'],bond['end_atom_idx'];parent[find(i)]=find(j)
        degree[i]+=1;degree[j]+=1;used[i]+=orders[bond['label_id']];used[j]+=orders[bond['label_id']]
        lengths.append(float(np.linalg.norm(centers[graph_to_raw[i]]-centers[graph_to_raw[j]])))
    audit['components_before']=len({find(i) for i in range(n)})
    if audit['components_before']==1:return [],audit
    from pair_training.notebook_ocr import group_text_targets
    blocked=set()
    for target in group_text_targets(graph,image):
        blocked.update(i for i in target['merged_atom_indices'] if i!=target['index'] and degree[i]==0)
    audit['protected_merged_text_fragments']=sorted(blocked)
    if lengths:median=float(np.median(lengths))
    else:
        d=np.linalg.norm(centers[:,None]-centers[None,:],axis=-1);np.fill_diagonal(d,np.inf);median=float(np.median(d.min(1)))
    proposals=[];coordinates=np.array([centers[graph_to_raw[i]] for i in range(n)])
    for pair,p in zip(pairs,probabilities):
        a,b=map(int,pair)
        if a not in raw_to_graph or b not in raw_to_graph:continue
        i,j=raw_to_graph[a],raw_to_graph[b]
        if i in blocked or j in blocked or find(i)==find(j):continue
        code=int(np.argmax(p[1:]))+1;category=code-1
        if category not in POLICY['allowed_types']:continue
        existence=float(1-p[0]);conditional=float(p[code]/max(existence,1e-12))
        if existence<1e-6 or conditional<POLICY['minimum_conditional_type']:continue
        start,end=centers[a],centers[b];delta=end-start;length=float(np.linalg.norm(delta))
        if not POLICY['minimum_length_ratio']*median<=length<=POLICY['maximum_length_ratio']*median:continue
        t=((coordinates-start)*delta).sum(1)/max(length*length,1e-6)
        distance=np.linalg.norm(coordinates-(start+t[:,None]*delta),axis=1)
        crossing=(t>.12)&(t<.88)&(distance<max(1.1,.06*length));crossing[[i,j]]=False
        if crossing.any():continue
        if used[i]+category>VALENCE.get(atoms[i]['symbol'],4) or used[j]+category>VALENCE.get(atoms[j]['symbol'],4):continue
        evidence=native_line_evidence(image,start,end,category)
        if evidence is None:continue
        score=existence if score_mode=='existence' else float(p[code])
        record=dict(begin=a,end=b,category_id=category,score=score,line=[*start.tolist(),*end.tolist()],
                    source='native_component_bridge',model_existence=existence,
                    conditional_type=conditional,native_evidence=evidence)
        proposals.append((evidence['continuity']+conditional,record,i,j))
    additions=[]
    for _,record,i,j in sorted(proposals,key=lambda x:x[0],reverse=True):
        order=record['category_id']
        if find(i)==find(j):continue
        if used[i]+order>VALENCE.get(atoms[i]['symbol'],4) or used[j]+order>VALENCE.get(atoms[j]['symbol'],4):continue
        parent[find(i)]=find(j);used[i]+=order;used[j]+=order;additions.append(record)
    audit.update(components_after=len({find(i) for i in range(n)}),added=additions)
    return additions,audit
