"""Atom-box to UNIQUE line-endpoint correspondence, with explicit ambiguity."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


def associate_unique_endpoints(boxes,lines,method='reverse_nearest',cluster_radius=0.):
    """Coordinates only: no atom/bond IDs, chemical labels or GT anchor inputs.

    reverse_nearest implements box-center -> endpoint KD-tree literally. A
    many-to-one collision is reported, never silently overwritten.
    global_center is an unconstrained Euclidean one-to-one assignment.
    exclusive_box locks mutually unique near-exact centers (0.01px rounding
    allowance), then matches remaining points to box rectangles one-to-one.
    Extra boxes can remain isolated; unique endpoints need not count all atoms.
    With cluster_radius=0, GT endpoints are deduplicated exactly. Positive
    radii enable bounded-diameter clustering for noisy model coordinates.
    exclusive_box_noisy additionally recovers nearby split clusters after
    establishing one-to-one box/representative-anchor correspondence.
    """
    boxes=np.asarray(boxes,float).reshape(-1,4);lines=np.asarray(lines,float).reshape(-1,2,2)
    if not np.isfinite(boxes).all() or not np.isfinite(lines).all():raise ValueError('Nonfinite geometry')
    points,inverse=np.unique(lines.reshape(-1,2),axis=0,return_inverse=True)
    raw_unique=len(points)
    if cluster_radius<0:raise ValueError('cluster_radius must be nonnegative')
    if cluster_radius and len(points):
        # Complete-diameter clusters avoid transitive chains joining distinct
        # atoms. The two different ends of one bond can never be merged.
        parent=np.arange(len(points));members={i:[i] for i in range(len(points))}
        incident={i:set() for i in range(len(points))}
        for bond,(a,b) in enumerate(inverse.reshape(-1,2)):
            incident[a].add(bond);incident[b].add(bond)
        def root(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]];i=parent[i]
            return i
        candidates=cKDTree(points).query_pairs(cluster_radius)
        for a,b in sorted(candidates,key=lambda ab:(np.linalg.norm(points[ab[0]]-points[ab[1]]),ab)):
            a,b=root(a),root(b)
            if a==b or incident[a]&incident[b]:continue
            if np.max(np.linalg.norm(points[members[a],None]-points[members[b]][None],axis=-1))>cluster_radius:continue
            parent[b]=a;members[a]+=members.pop(b);incident[a]|=incident.pop(b)
        roots=sorted(members);lookup={r:i for i,r in enumerate(roots)}
        inverse=np.array([lookup[root(int(i))] for i in inverse])
        points=np.array([points[members[r]].mean(0) for r in roots])
    n=len(boxes);m=len(points);centers=boxes[:,:2]+boxes[:,2:]/2
    owners=np.full(m,-1,int);box_to_endpoint=np.full(n,-1,int);locked=0
    if n and m:
        distance,nearest=cKDTree(points).query(centers,k=1)
        counts=np.bincount(nearest,minlength=m)
        if method=='reverse_nearest':
            box_to_endpoint=nearest.copy()
            for box_index,endpoint in enumerate(nearest):
                if counts[endpoint]==1:owners[endpoint]=box_index
        elif method in ['global_center','exclusive_box','exclusive_box_noisy']:
            remaining_boxes=np.arange(n);remaining_points=np.arange(m)
            if method in ['exclusive_box','exclusive_box_noisy']:
                # Both directions must agree and the candidate must be a unique
                # near-exact coordinate match; no arbitrary tie/ID preference.
                back_distance,back=cKDTree(centers).query(points,k=min(2,n))
                back=np.asarray(back).reshape(m,min(2,n));back_distance=np.asarray(back_distance).reshape(m,min(2,n))
                for i,j in enumerate(nearest):
                    unique=n==1 or back_distance[j,1]-back_distance[j,0]>1e-9
                    if distance[i]<=.01 and back[j,0]==i and unique:
                        owners[j]=i;box_to_endpoint[i]=j;locked+=1
                remaining_boxes=np.flatnonzero(box_to_endpoint<0);remaining_points=np.flatnonzero(owners<0)
            if len(remaining_boxes) and len(remaining_points):
                b=boxes[remaining_boxes];p=points[remaining_points]
                center_distance=np.linalg.norm(centers[remaining_boxes,None]-p[None],axis=-1)
                if method in ['exclusive_box','exclusive_box_noisy']:
                    boundary=np.linalg.norm(np.maximum(np.maximum(b[:,None,:2]-p[None],p[None]-(b[:,:2]+b[:,2:])[:,None]),0),axis=-1)
                    cost=boundary+.02*center_distance
                else:cost=center_distance
                rows,cols=linear_sum_assignment(cost)
                for i,j in zip(remaining_boxes[rows],remaining_points[cols]):owners[j]=i;box_to_endpoint[i]=j
        else:raise ValueError(method)
    else:
        counts=np.zeros(m,int)
    recovered=0
    if method=='exclusive_box_noisy' and n and m:
        # A predicted atom contributes one endpoint per incident bond. Noise
        # may split those repeated coordinates into multiple clusters. Keep
        # the one-to-one anchor assignment, then attach only nearby surplus
        # clusters to established anchors, gated by both box and anchor geometry.
        # This path is never used when all unique GT endpoints are assigned.
        missing=np.flatnonzero(owners<0);anchors=np.flatnonzero(owners>=0)
        if len(missing) and len(anchors):
            length=np.linalg.norm(lines[:,1]-lines[:,0],axis=1)
            positive=length[length>1e-6]
            gate=float(np.clip(.35*np.median(positive),3,12)) if len(positive) else 3.
            p=points[missing];a=points[anchors];b=boxes[owners[anchors]]
            distance=np.linalg.norm(p[:,None]-a[None],axis=-1)
            boundary=np.linalg.norm(np.maximum(np.maximum(b[None,:,:2]-p[:,None],p[:,None]-(b[:,:2]+b[:,2:])[None]),0),axis=-1)
            cost=np.where((distance<=gate)&(boundary<=gate),boundary+.15*distance,np.inf)
            winner=cost.argmin(1)
            for row,j in enumerate(winner):
                if np.isfinite(cost[row,j]):owners[missing[row]]=owners[anchors[j]];recovered+=1
    endpoint_pairs=inverse.reshape(-1,2);pairs=[]
    for a,b in endpoint_pairs:
        i,j=owners[a],owners[b]
        pairs.append((int(i),int(j)) if min(i,j)>=0 and i!=j else None)
    return pairs,dict(method=method,atom_boxes=n,unique_endpoints=m,
        raw_unique_endpoints=raw_unique,cluster_radius=float(cluster_radius),
        recovered_split_endpoint_clusters=recovered,
        endpoint_count_matches_atoms=bool(n==m),
        raw_nearest_collision_endpoints=int((counts>1).sum()),raw_nearest_uncovered_endpoints=int((counts==0).sum()),
        unresolved_endpoints=int((owners<0).sum()),boxes_without_endpoint=int((box_to_endpoint<0).sum()),
        locked_exact_centers=locked,box_to_endpoint=box_to_endpoint.tolist(),endpoint_to_box=owners.tolist(),
        unique_endpoint_coordinates=points.tolist())
