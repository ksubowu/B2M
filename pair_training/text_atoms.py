"""OCR-supported whole-label selection before atom-pair inference (notebook path)."""
import math
import numpy as np
import torch
import string
from torchvision.ops import box_convert
from pair_training.notebook_ocr import _chem_helpers, paddle_recognizer


def accepted_label(chosen,min_confidence=.85,allow_unmapped=False):
    if not math.isfinite(chosen['score']) or not min_confidence<=chosen['score']<=1:return False
    if chosen['label_type']=='abbreviation':return True
    if not allow_unmapped or chosen['label_type'] not in ('unmapped','rgroup'):return False
    text=chosen['raw_text']
    return len(text)>=2 and set(text)<=set(string.ascii_letters+string.digits)


def detect_label_regions(image, scales=(1., 2.), min_confidence=.85,recognizer=None,
                         detector_only=False,allow_unmapped=False):
    """Native-raster text detection plus multi-scale recognition of whole regions."""
    from engine.rtv4.chem_ocr_expand import _get_paddleocr_instances
    from pair_training.notebook_ocr import recognize_multiscale
    primary,_=_get_paddleocr_instances(use_gpu=False)
    if primary is None:raise RuntimeError('PaddleOCR不可用，不能进行文本区域整框选择')
    # PaddleOCR 2.x's DB detector still references this removed NumPy alias.
    if not hasattr(np,'int'):np.int=int
    regions=[]
    for scale in scales:
        raster=image.convert('RGB').resize((round(image.width*scale),round(image.height*scale)))
        # Detector-only proposals avoid dropping a complete label merely because
        # the generic recognizer could not read it before the adapted reader.
        result=(primary.ocr(np.asarray(raster),det=True,rec=False,cls=False) if detector_only
                else primary.ocr(np.asarray(raster),cls=True)) or []
        entries=result[0] if len(result)==1 else result
        for entry in entries or []:
            try:
                if detector_only:quad=entry;text='';score=0.
                else:quad,(text,score)=entry
                points=np.asarray(quad,dtype=float)/scale
                region=dict(box=[float(points[:,0].min()),float(points[:,1].min()),
                                 float(points[:,0].max()),float(points[:,1].max())],
                            detector_text=str(text),detector_recognition_score=float(score),detection_scale=scale)
                # Whole-region candidates remain available; recognition below
                # uses the SAME OCR engine and confidence selection as crops.
                graph={'atoms':[dict(index=0,symbol='*',box=region['box'])],'bonds':[]}
                labels,candidates=recognize_multiscale(graph,image,min_confidence=min_confidence,recognizer=recognizer)
                if not labels:continue
                chosen=labels[0]
                region.update(text=chosen['text'],raw_text=chosen['raw_text'],score=chosen['score'],
                              label_type=chosen['label_type'],recognition_candidates=candidates,
                              recognition_box=candidates[0]['recognition_box'])
                region['accepted_text']=accepted_label(chosen,min_confidence,allow_unmapped)
                region['reader_backend']='custom' if recognizer else 'paddle'
                regions.append(region)
            except (TypeError,ValueError,IndexError) as exc:
                regions.append(dict(error=str(exc),accepted_text=False))
    return regions


def _real_line_in_gap(image, a, b):
    """Measure a thin continuous ink corridor between boxes at any angle.

    Glyph strokes occupy many transverse rows; a real bond occupies a thin
    corridor after rotating the sampling grid onto the center-to-center line.
    """
    aa=np.asarray([(a[0]+a[2])/2,(a[1]+a[3])/2],dtype=float)
    bb=np.asarray([(b[0]+b[2])/2,(b[1]+b[3])/2],dtype=float)
    delta=bb-aa;length=float(np.linalg.norm(delta))
    if length<3:return False
    axis=delta/length;normal=np.asarray([-axis[1],axis[0]])
    center=aa+np.linspace(0,1,max(8,math.ceil(length*2)))[:,None]*delta
    def outside(points,box):
        return ((points[...,0]<box[0]-.5)|(points[...,0]>box[2]+.5)|
                (points[...,1]<box[1]-.5)|(points[...,1]>box[3]+.5))
    center=center[outside(center,a)&outside(center,b)]
    if len(center)<6:return False
    h=max(4.,min(a[3]-a[1],b[3]-b[1]))
    offsets=np.linspace(-h/2,h/2,max(7,math.ceil(h)+1))
    grid=center[None,:,:]+offsets[:,None,None]*normal
    valid=outside(grid,a)&outside(grid,b)
    x=np.rint(grid[...,0]).astype(int);y=np.rint(grid[...,1]).astype(int)
    valid&=(x>=0)&(x<image.width)&(y>=0)&(y<image.height)
    ink=np.asarray(image.convert('RGB')).min(axis=2)<180
    sampled=ink[np.clip(y,0,image.height-1),np.clip(x,0,image.width-1)]&valid
    center_rows=np.abs(offsets)<=1.5
    coverage=(sampled[center_rows].any(axis=0)).mean()
    thin=sampled.any(axis=1).sum()<=max(4,len(offsets)*.45)
    return bool(coverage>=.85 and thin)


def reconcile_text_atoms(atoms, image, regions):
    """Prefer a validated complete text region over its constituent detections.

    Detection scores do not compete with OCR scores: OCR authorizes a semantic
    whole-label region, while the atom score remains max supporting atom score.
    No blanket large-box preference. No atom is deleted across a real bond gap.
    Returns a new atom record and a complete reconciliation audit.
    """
    device=atoms['centers'].device
    scale=atoms['centers'].new_tensor([image.width,image.height])
    pixel_boxes=box_convert(atoms['boxes'],'cxcywh','xyxy').detach().cpu()*torch.tensor([image.width,image.height]*2)
    centers=(atoms['centers']*scale).detach().cpu().tolist()
    removed=set(); additions=[]; audit=[]
    # Overlapping duplicate OCR detections are handled by coverage of atom IDs.
    for region in sorted(regions,key=lambda r:-r.get('score',0)):
        row={k:v for k,v in region.items() if k!='recognition_candidates'}
        row['action']='rejected'
        if not region.get('accepted_text'):
            row['reason']='not_a_supported_confident_abbreviation';audit.append(row);continue
        box=region['recognition_box']
        members=[i for i,(x,y) in enumerate(centers) if box[0]-1<=x<=box[2]+1 and box[1]-1<=y<=box[3]+1]
        if not members:
            row['reason']='no_supporting_atom_detection';audit.append(row);continue
        if removed.intersection(members):
            row['reason']='overlapping_region_already_selected';audit.append(row);continue
        height=box[3]-box[1]
        if height<3 or box[2]-box[0]>height*12:
            row['reason']='invalid_text_geometry';audit.append(row);continue
        if any(_real_line_in_gap(image,pixel_boxes[i].tolist(),pixel_boxes[j].tolist())
               for i,j in zip(sorted(members,key=lambda i:centers[i][0]),sorted(members,key=lambda i:centers[i][0])[1:])):
            row['reason']='printed_bond_between_atoms';audit.append(row);continue
        # A region crossing several rows of atoms is not a single printed word.
        if max(centers[i][1] for i in members)-min(centers[i][1] for i in members)>height*.55:
            row['reason']='multiple_atom_rows';audit.append(row);continue
        x1,y1,x2,y2=box
        additions.append(dict(center=[(x1+x2)/2/image.width,(y1+y2)/2/image.height],
                              box=[(x1+x2)/2/image.width,(y1+y2)/2/image.height,
                                   (x2-x1)/image.width,(y2-y1)/image.height],
                              score=max(float(atoms['scores'][i]) for i in members),
                              text=region['raw_text'],members=members))
        removed.update(members)
        row.update(action='selected',reason='whole_text_over_fragments',replaced_atom_indices=members,
                   replaced_labels=[int(atoms['labels'][i]) for i in members],
                   replaced_scores=[float(atoms['scores'][i]) for i in members])
        audit.append(row)
    keep=[i for i in range(len(centers)) if i not in removed]
    result={k:v for k,v in atoms.items() if k not in ('centers','boxes','scores','labels')}
    for key,width in [('centers',2),('boxes',4)]:
        extra=[a['center' if key=='centers' else 'box'] for a in additions]
        result[key]=torch.cat((atoms[key][keep],atoms[key].new_tensor(extra).reshape(-1,width)),0)
    result['labels']=torch.cat((atoms['labels'][keep],atoms['labels'].new_zeros(len(additions))))
    result['scores']=torch.cat((atoms['scores'][keep],atoms['scores'].new_tensor([a['score'] for a in additions])))
    return result,dict(original_atoms=len(centers),selected_atoms=len(result['centers']),regions=audit,
                       unchanged_original_indices=keep,whole_text_atoms=additions)


@torch.no_grad()
def apply_text_anchors(atoms,image,features,head):
    """Move only text-like wildcard sampling points; keep entity boxes/classes.

    No GT is used. Clamp regressed points to the text box plus 2 native pixels.
    This preserves the geometry policy used in the controlled structure pilot.
    """
    xyxy=box_convert(atoms['boxes'],'cxcywh','xyxy').clamp(0,1)
    size=xyxy.new_tensor(image.size)
    dimensions=(xyxy[:,2:]-xyxy[:,:2])*size
    ids=torch.where((atoms['labels']==0)&(dimensions[:,0]>1.2*dimensions[:,1])&(dimensions[:,1]>=4))[0]
    result={**atoms,'centers':atoms['centers'].clone()}
    if len(ids):
        native=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).to(xyxy.device).float()/255
        _,points=head(features,[xyxy[ids]],images=[native],return_anchors=True)
        margin=2/size
        points=torch.maximum(xyxy[ids,:2]-margin,torch.minimum(points,xyxy[ids,2:]+margin)).clamp(0,1)
        result['centers'][ids]=points
    audit=dict(atom_indices=ids.cpu().tolist(),
               old_points=(atoms['centers'][ids]*size).cpu().tolist(),
               new_points=(result['centers'][ids]*size).cpu().tolist(),policy='wildcard aspect>1.2 height>=4px; box plus 2px clamp')
    return result,audit


def decode_reference(atom, pairs, logits, image_size, threshold, score_mode):
    """Unmodified CLI decision rule, for raw-versus-text-aware comparisons."""
    from pair_training.model import decode_bond
    coordinates=(atom['centers'].detach().cpu()*torch.tensor(image_size)).tolist()
    records=[dict(index=i,center=point,category_id=int(atom['labels'][i]),score=float(atom['scores'][i]))
             for i,point in enumerate(coordinates)]
    bonds=[]
    for pair,prob in zip(pairs.cpu().tolist(),logits.softmax(-1).detach().cpu()):
        score,label=prob[1:].max(0);label=int(label)+1
        selection=1-prob[0] if score_mode=='existence' else score
        if float(selection)<threshold or selection<=prob[0]:continue
        category,reverse=decode_bond(label)
        a,b=pair[::-1] if reverse else pair
        bonds.append(dict(begin=a,end=b,category_id=category,score=float(selection),
                          type_probability_given_bond=float(score/(1-prob[0]).clamp_min(1e-9)),
                          none_score=float(prob[0]),line=coordinates[a]+coordinates[b]))
    return dict(atoms=records,bonds=bonds,candidate_pairs=len(pairs),threshold=threshold)
