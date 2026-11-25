"""Shared-query line/band pairing, line-only retention, and graph uniqueness."""
import numpy as np
from .geometry import assemble_band_lines,LineEndpointIndex


def refine_peak_lines(atom_boxes,decoded,line_threshold=.2,minimum_improvement=1.):
    """One geometry hypothesis per accepted peak, using its 3x3 neighborhood.

    Presence/query identity remain those of the original line peak. Geometry
    may come from a nearby line-head cell, never from band scores or GT. This
    requires pre-peak candidates from decode(..., apply_line_peaks=False).
    """
    from scipy.spatial import cKDTree
    raw=decoded['lines'];peaks=[r for r in raw if r.get('is_local_peak',True) and r['line_score']>=line_threshold]
    if not atom_boxes or not raw:return dict(decoded,lines=peaks)
    boxes=np.asarray(atom_boxes,float);centers=boxes[:,:2]+boxes[:,2:]/2
    points=np.asarray([r['line'] for r in raw]);dist,owner=cKDTree(centers).query(points.reshape(-1,2))
    assigned=boxes[owner].reshape(-1,2,4)
    boundary=np.linalg.norm(np.maximum(np.maximum(assigned[:,:,:2]-points,points-(assigned[:,:,:2]+assigned[:,:,2:])),0),axis=-1)
    quality=(boundary+.02*dist.reshape(-1,2)).mean(1)
    index={r['query_id']:i for i,r in enumerate(raw)};width=decoded.get('grid_width',160);result=[]
    for peak in peaks:
        q=peak['query_id'];y,x=divmod(q,width);allowed=[]
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                if not 0<=x+dx<width or y+dy<0:continue
                i=index.get((y+dy)*width+x+dx)
                if i is not None and raw[i]['line_score']>=max(line_threshold,.5*peak['line_score']):allowed.append(i)
        best=min(allowed,key=lambda i:(quality[i],-raw[i]['line_score']))
        if quality[index[q]]-quality[best]>=minimum_improvement:
            chosen=raw[best];result.append(dict(peak,line=chosen['line'],geometry_query_id=chosen['query_id'],
                geometry_score=chosen['line_score'],geometry_improvement=float(quality[index[q]]-quality[best])))
        else:result.append(peak)
    return dict(decoded,lines=result)


def group_lines(lines,radius=3.):
    groups=[]
    for item in sorted(lines,key=lambda r:-r['line_score']):
        segment=np.asarray(item['line']);length=float(np.linalg.norm(segment[1]-segment[0]));matched=False
        for group in groups:
            other=np.asarray(group[0]['line']);other_length=float(np.linalg.norm(other[1]-other[0]))
            distance=min(np.linalg.norm(segment-other,axis=1).max(),np.linalg.norm(segment-other[::-1],axis=1).max())
            if distance<=min(radius,max(.5,.12*min(length,other_length))):
                group.append(item);matched=True;break
        if not matched:groups.append([item])
    return groups


def assemble_scored_bonds(atom_boxes,decoded,line_nms_radius=3.,line_association='forward',endpoint_cluster_radius=0.,stereo_from_band=False):
    by_query={b['query_id']:b for b in decoded['bands']+decoded.get('paired_band_probes',[])};instances=[];untyped=[];used=set()
    groups=group_lines(decoded['lines'],line_nms_radius)
    representatives=[max(group,key=lambda r:r['line_score']) for group in groups]
    if line_association=='forward':
        pairs=LineEndpointIndex(atom_boxes).link_many([r['line'] for r in representatives]);assignment_audit=None
    else:
        from .endpoint_assignment import associate_unique_endpoints
        pairs,assignment_audit=associate_unique_endpoints(atom_boxes,[r['line'] for r in representatives],method=line_association,cluster_radius=endpoint_cluster_radius)
    # Geometrically separate line hypotheses can still map to the same atom
    # pair. Merge their evidence before selecting geometry and type separately.
    merged={}
    for i,(group,line,pair) in enumerate(zip(groups,representatives,pairs)):
        key=tuple(sorted(pair)) if pair is not None else ('unlinked',i)
        if key not in merged:merged[key]=dict(representatives=[],queries=[],pairs={})
        merged[key]['representatives'].append(line);merged[key]['queries']+=group
        merged[key]['pairs'][line['query_id']]=pair
    line_edges=[];unlinked_lines=[];removed=[];instance_pairs=[];stereo_flips=[]
    for merged_group in merged.values():
        line=max(merged_group['representatives'],key=lambda r:r['line_score'])
        pair=merged_group['pairs'][line['query_id']];group=merged_group['queries']
        removed += [r['query_id'] for r in group if r['query_id']!=line['query_id']]
        if pair is None:unlinked_lines.append(dict(line))
        else:line_edges.append(dict(begin=pair[0],end=pair[1],line=line['line'],line_score=line['line_score'],query_id=line['query_id']))
        candidates=[by_query[r['query_id']] for r in group if r['query_id'] in by_query]
        if not candidates:
            untyped.append(dict(line,atom_pair=list(pair) if pair is not None else None));continue
        band=max(candidates,key=lambda r:r['score']);used.add(band['query_id'])
        if stereo_from_band and band['label'] in [4,5]:
            quad=np.asarray(band['quad']);segment=np.asarray(line['line'])
            if np.dot(quad[2:].mean(0)-quad[:2].mean(0),segment[1]-segment[0])<0:
                # Directed training quads carry a second orientation signal.
                # Only reverse endpoint order; never alter endpoint positions.
                line=dict(line,line=segment[::-1].tolist());stereo_flips.append(line['query_id'])
                if pair is not None:
                    pair=pair[::-1];line_edges[-1]=dict(line_edges[-1],begin=pair[0],end=pair[1],line=line['line'])
        instances.append(dict(label=band['label'],quad=band['quad'],line=line['line'],score=band['score'],
            line_score=line['line_score'],band_score=band['band_score'],type_score=band['type_score'],
            line_query=line['query_id'],band_query=band['query_id'],line_group_queries=[r['query_id'] for r in group]))
        instance_pairs.append(pair)
    graph=assemble_band_lines(atom_boxes,instances,line_association='forward')
    if line_association!='forward':
        # Preserve the assignment made with ALL retained lines, including
        # untyped ones. Re-solving only typed lines can change correspondence.
        names=['single','double','triple','aromatic','wedge','dash']
        graph['bonds']=[dict(bond_instance=i,begin=p[0],end=p[1],label=int(b['label']),type=names[b['label']],score=float(b['score']))
                        for i,(b,p) in enumerate(zip(instances,instance_pairs)) if p is not None]
        graph['unlinked_band_indices']=[i for i,p in enumerate(instance_pairs) if p is None]
        graph['line_association']=line_association;graph['endpoint_assignment_audit']=assignment_audit
    keys=[tuple(sorted((e['begin'],e['end']))) for e in graph['bonds']]
    assert len(keys)==len(set(keys)),'Only one chemical bond per atom pair'
    graph.update(instances=instances,line_graph=line_edges,unlinked_lines=unlinked_lines,untyped_lines=untyped,band_only_candidates=[b for b in decoded['bands'] if b['query_id'] not in used],
        removed_duplicate_line_queries=removed,stereo_band_direction_flips=stereo_flips,raw_line_candidates=len(decoded['lines']),raw_band_candidates=len(decoded['bands']))
    for edge in graph['bonds']:
        instance=instances[edge['bond_instance']]
        edge.update(line_score=instance['line_score'],band_score=instance['band_score'],type_score=instance['type_score'])
    return graph
