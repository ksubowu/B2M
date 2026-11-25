"""Geometry-only graph assembly; atom indices appear only in evaluation."""
import itertools
from collections import Counter
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def polygon_iou(a,b):
    a=cv2.convexHull(np.asarray(a,np.float32));b=cv2.convexHull(np.asarray(b,np.float32))
    area_a=cv2.contourArea(a);area_b=cv2.contourArea(b)
    if min(area_a,area_b)<1e-4:return 0.
    overlap=cv2.intersectConvexConvex(a,b)[0]
    return float(overlap/max(1e-6,area_a+area_b-overlap))


def suppress(predictions):
    kept=[];bounds=[]
    for p in sorted(predictions,key=lambda p:-p['score']):
        points=np.asarray(p['quad']);low=points.min(0);high=points.max(0)
        duplicate=False
        for q,(qlow,qhigh) in zip(kept,bounds):
            # Disjoint bounding rectangles imply zero polygon intersection.
            if np.any(high<=qlow) or np.any(qhigh<=low):continue
            if polygon_iou(p['quad'],q['quad'])>=.45:duplicate=True;break
        if not duplicate:kept.append(p);bounds.append((low,high))
    return kept


def link(quad,boxes):
    """Associate region ends by boundary distance and axis compatibility.

    No pre-generated atom-pair proposals; ambiguous ends are rejected. Boxes are
    xywh. The caller supplies either predictions or oracle boxes for diagnostics.
    """
    if len(boxes)<2:return None
    q=np.asarray(quad,float);ends=np.stack((q[:2].mean(0),q[2:].mean(0)))
    center=ends.mean(0);delta=ends[1]-ends[0];length=np.linalg.norm(delta)
    if length<3:return None
    axis=delta/length;normal=np.array([-axis[1],axis[0]])
    candidates=[]
    for side,end in enumerate(ends):
        scores=[]
        for i,box in enumerate(boxes):
            x,y,w,h=box;bc=np.array([x+w/2,y+h/2])
            # A bond end cannot attach to an atom near the band's middle.
            along=np.dot(bc-center,axis)*(1 if side else -1)
            if along<length*.18:continue
            distance=np.linalg.norm(np.maximum(np.maximum(np.array([x,y])-end,end-np.array([x+w,y+h])),0))
            perp=max(0,abs(np.dot(bc-end,normal))-(abs(normal[0])*w+abs(normal[1])*h)/2)
            if distance>max(13.,length*.55) or perp>max(7.,length*.22):continue
            score=distance+.35*perp+.05*np.linalg.norm(bc-end)
            scores.append((score,i))
        scores.sort()
        candidates.append(scores[:3])
    pairs=sorted((a[0]+b[0],a[1],b[1]) for a,b in itertools.product(*candidates) if a[1]!=b[1])
    if not pairs:return None
    if len(pairs)>1 and pairs[1][0]-pairs[0][0]<1.:return None
    return pairs[0][1:]


class AtomCenterIndex:
    """One atom-center KD-tree per image; query all band ends in one batch.

    Box centers are used, not annotated atom anchor coordinates. The two ends
    retain their order, including tip-to-wide ordering for wedge/dash bonds.
    No angle/box-boundary scoring or nearest-neighbor pair enumeration is used.
    A same-atom connection is rejected; it is not silently changed to k=2.

    Reference: i2m_data/real_eval_scripts_only_20261111/det_engine.py,
    bbox2center() and evaluate_x()'s KDTree/anchor_positions query near line 4849.
    That implementation needs two diagonal hypotheses for axis-aligned bond
    boxes. Ordered band quadrilaterals already identify their two ends, so no
    diagonal search or later bond-center k=2 rematching is needed here.
    The reference uses xyxy boxes; this module explicitly consumes xywh boxes.
    """
    def __init__(self, boxes):
        boxes=np.asarray(boxes,dtype=float).reshape(-1,4)
        self.centers=boxes[:,:2]+boxes[:,2:]/2
        self.tree=cKDTree(self.centers) if len(boxes)>=2 else None

    def link_many(self, quads, max_distance=None):
        quads=np.asarray(quads,dtype=float).reshape(-1,4,2)
        if self.tree is None:return [None]*len(quads)
        if not len(quads):return []
        ends=np.stack((quads[:,:2].mean(1),quads[:,2:].mean(1)),axis=1)
        valid=np.isfinite(ends).all(axis=(1,2))
        valid &= np.linalg.norm(ends[:,1]-ends[:,0],axis=1)>=3
        safe=np.where(valid[:,None,None],ends,0.)
        distances,indices=self.tree.query(safe.reshape(-1,2),k=1)
        indices=indices.reshape(-1,2);distances=distances.reshape(-1,2)
        valid &= indices[:,0]!=indices[:,1]
        if max_distance is not None:valid &= (distances<=max_distance).all(1)
        return [(int(a),int(b)) if ok else None for (a,b),ok in zip(indices,valid)]


class DirectionalBandIndex:
    """Directional continuation from ordered band ends to atom-box anchors.

    Long labels have multiple spatial anchors mapped to ONE original atom,
    following I2M/evaluate/eval_model.py:preprocess_atom_boxes. The band axis
    removes the diagonal ambiguity of bond boxes. No GT identities, degrees,
    atom labels, valence rules or annotated atom centers are used.
    """
    def __init__(self,boxes):
        boxes=np.asarray(boxes,float).reshape(-1,4);points=[];owners=[]
        self.atom_count=len(boxes)
        for i,(x,y,w,h) in enumerate(boxes):
            if not np.isfinite([x,y,w,h]).all() or min(w,h)<=0:continue
            n=min(32,int(np.ceil(max(w,h)/min(w,h)))) if max(w,h)/min(w,h)>1.5 else 1
            anchors=np.tile([x+w/2,y+h/2],(n,1));dim=0 if w>=h else 1
            anchors[:,dim]=(x if dim==0 else y)+(np.arange(n)+.5)/n*(w if dim==0 else h)
            points.extend(anchors);owners.extend([i]*n)
        self.points=np.asarray(points,float).reshape(-1,2);self.owners=np.asarray(owners,int)
        self.tree=cKDTree(self.points) if len(points) and len(boxes)>=2 else None

    def link_many(self,quads):
        q=np.asarray(quads,float).reshape(-1,4,2)
        if self.tree is None:return [None]*len(q)
        ends=np.stack((q[:,:2].mean(1),q[:,2:].mean(1)),1)
        axes=ends[:,1]-ends[:,0];length=np.linalg.norm(axes,axis=1)
        valid=np.isfinite(q).all((1,2))&(length>=3)
        axes=axes/np.maximum(length[:,None],1e-9);mid=ends.mean(1)
        width=np.stack((np.linalg.norm(q[:,0]-q[:,1],axis=1),np.linalg.norm(q[:,2]-q[:,3],axis=1)),1)
        positions=ends.reshape(-1,2);directions=(axes[:,None,:]*np.array([-1,1])[None,:,None]).reshape(-1,2)
        middles=np.repeat(mid,2,axis=0);slack=np.maximum(2,width*.25).reshape(-1)
        best=np.full(len(positions),np.inf);answer=np.full(len(positions),-1,int)
        active=np.flatnonzero(np.repeat(valid,2));k=min(8,len(self.points))
        # All penalties are nonnegative. Expand KD candidates until the next
        # Euclidean distance cannot beat the best directional score. Thus this
        # returns the same optimum as testing every anchor, without a fixed k cap.
        while len(active):
            distances,idx=self.tree.query(positions[active],k=k)
            distances=np.asarray(distances).reshape(len(active),k);idx=np.asarray(idx).reshape(len(active),k)
            p=self.points[idx];delta=p-positions[active,None,:]
            outward=(delta*directions[active,None,:]).sum(-1)
            correct_side=((p-middles[active,None,:])*directions[active,None,:]).sum(-1)>0
            perp=np.abs(delta[:,:,0]*directions[active,None,1]-delta[:,:,1]*directions[active,None,0])
            score=distances+2*perp+2*np.maximum(0,-outward-slack[active,None])
            score=np.where(correct_side,score,np.inf)
            choice=score.argmin(1);best[active]=score[np.arange(len(active)),choice]
            answer[active]=self.owners[idx[np.arange(len(active)),choice]]
            if k==len(self.points):break
            active=active[distances[:,-1]<=best[active]];k=min(len(self.points),k*2)
        answer=answer.reshape(-1,2);best=best.reshape(-1,2)
        valid &= np.isfinite(best).all(1)&(answer[:,0]!=answer[:,1])
        return [(int(a),int(b)) if ok else None for (a,b),ok in zip(answer,valid)]


def link_many(quads, boxes, method='geometry'):
    """Shared assembly dispatch. KD-tree construction is amortized per image."""
    if method=='kdtree':return AtomCenterIndex(boxes).link_many(quads)
    if method=='band_directional':return DirectionalBandIndex(boxes).link_many(quads)
    if method=='geometry':return [link(q,boxes) for q in quads]
    raise ValueError('Unknown association method: '+method)


class LineEndpointIndex(AtomCenterIndex):
    """One KD-tree over atom-box centers; ordered center-to-center line ends.

    Unlike visible band ink, the complete line represents atom connections.
    No minimum visible-stroke length, taper/axis scoring or band-end snapping
    applies. Atom centers here always come from input boxes, not GT anchors.
    """
    def link_many(self,lines,max_distance=None):
        points=np.asarray(lines,float).reshape(-1,2,2)
        if self.tree is None:return [None]*len(points)
        if not len(points):return []
        valid=np.isfinite(points).all((1,2));safe=np.where(valid[:,None,None],points,0.)
        distances,indices=self.tree.query(safe.reshape(-1,2),k=1)
        distances=distances.reshape(-1,2);indices=indices.reshape(-1,2)
        valid &= indices[:,0]!=indices[:,1]
        if max_distance is not None:valid &= (distances<=max_distance).all(1)
        return [(int(a),int(b)) if ok else None for (a,b),ok in zip(indices,valid)]


def assemble_band_lines(atom_boxes,bond_instances,max_endpoint_distance=None,line_association='forward',endpoint_cluster_radius=0.):
    """Geometry-only graph from paired band+line outputs for EACH bond instance.

    atom_boxes: xywh pixels. Each instance: label (0..5), quad (4x2), line (2x2).
    Pairing must be supplied by the same bond query/record, not inferred from
    independently shuffled decoder outputs. GT source/target IDs are neither
    accepted nor required. Band supplies class/region; line supplies connection
    and preserves its original endpoint order, including wedge/dash direction.
    endpoint_cluster_radius defaults to zero for exact GT correspondence;
    prediction callers may explicitly enable tolerance in the same pixel frame.
    """
    names=['single','double','triple','aromatic','wedge','dash']
    lines=[];bands=[]
    for instance in bond_instances:
        if not all(key in instance for key in ['label','quad','line']):
            raise ValueError('Each bond instance requires paired label, quad and line')
        label=instance['label']
        if label not in range(6):raise ValueError('Unsupported band label')
        quad=np.asarray(instance['quad'],float).reshape(4,2)
        if not np.isfinite(quad).all():raise ValueError('Nonfinite band geometry')
        lines.append(np.asarray(instance['line'],float).reshape(2,2))
        bands.append(dict(label=int(label),quad=quad.tolist()))
    if line_association=='forward':
        pairs=LineEndpointIndex(atom_boxes).link_many(lines,max_distance=max_endpoint_distance);assignment_audit=None
    else:
        from .endpoint_assignment import associate_unique_endpoints
        pairs,assignment_audit=associate_unique_endpoints(atom_boxes,lines,method=line_association,cluster_radius=endpoint_cluster_radius)
    edges=[];unlinked=[]
    for i,(instance,pair) in enumerate(zip(bond_instances,pairs)):
        if pair is None:unlinked.append(i);continue
        edges.append(dict(bond_instance=i,begin=pair[0],end=pair[1],label=int(instance['label']),
                          type=names[instance['label']],score=float(instance.get('score',1.))))
    return dict(association='band_line',bonds=edges,unlinked_band_indices=unlinked,
                bands=bands,lines=[v.tolist() for v in lines],line_association=line_association,
                endpoint_assignment_audit=assignment_audit)


def edge_key(a,b,label):
    return (a,b,label) if label in (4,5) else (*sorted((a,b)),label)


def target_edges(annotation):
    return Counter(edge_key(b['begin'],b['end'],b['category_id']-1) for b in annotation['bonds'])


def pred_edges(predictions,boxes,mapping=None,method='geometry'):
    edges=Counter();unlinked=0
    links=link_many([p['quad'] for p in predictions],boxes,method)
    for p,ends in zip(predictions,links):
        if ends is None:unlinked+=1;continue
        a,b=ends
        if mapping is not None:
            a=mapping.get(a,-1);b=mapping.get(b,-1)
            if a<0 or b<0:unlinked+=1;continue
        edges[edge_key(a,b,p['label'])]+=1
    return edges,unlinked


def edge_metrics(predictions,annotation,boxes,mapping=None,method='geometry'):
    truth=target_edges(annotation);pred,unlinked=pred_edges(predictions,boxes,mapping,method)
    tp=sum((truth&pred).values());total_pred=len(predictions);total_gt=sum(truth.values())
    return dict(tp=tp,fp=total_pred-tp,fn=total_gt-tp,unlinked=unlinked,
                exact=int(pred==truth and unlinked==0),gt=total_gt)


def atom_mapping(predictions,annotation):
    if not predictions or not annotation['atoms']:return {},0
    pts=np.asarray([[p['bbox'][0]+p['bbox'][2]/2,p['bbox'][1]+p['bbox'][3]/2] for p in predictions])
    gt=np.asarray([a['center'] for a in annotation['atoms']])
    dist=np.linalg.norm(pts[:,None]-gt[None],axis=2)
    labels=np.asarray([p['label'] for p in predictions]);gl=np.asarray([a['category_id'] for a in annotation['atoms']])
    cost=dist+(labels[:,None]!=gl[None])*1000
    rows,cols=linear_sum_assignment(cost)
    mapping={int(i):int(j) for i,j in zip(rows,cols) if cost[i,j]<14}
    return mapping,len(mapping)


def aggregate(rows):
    total={key:sum(r[key] for r in rows) for key in ['tp','fp','fn','unlinked','exact','gt']}
    total.update(precision=total['tp']/max(1,total['tp']+total['fp']),
                 recall=total['tp']/max(1,total['tp']+total['fn']),
                 f1=2*total['tp']/max(1,2*total['tp']+total['fp']+total['fn']),
                 graph_exact=total['exact']/max(1,len(rows)),images=len(rows))
    return total
