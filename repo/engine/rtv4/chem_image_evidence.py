"""Image-only bond support diagnostics, without ground-truth information.

Carbon vertices usually lie on drawing ink. Explicit element glyph centers may
be white, and double/dashed bonds need not have ink on their centerline. Endpoint
extensions are accepted when nearby detected strokes align with the candidate.
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates


class BondImageEvidence:
    def __init__(self,image,angle_limit=15.,minimum_body_support=None):
        rgb=np.asarray(image.convert('RGB'))
        self.distance=distance_transform_edt(rgb.min(axis=2)>180)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        detected=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
        self.strokes=detected[:,0].astype(float) if detected is not None else np.zeros((0,4))
        delta=self.strokes[:,2:]-self.strokes[:,:2]
        self.stroke_lengths=np.linalg.norm(delta,axis=1)
        self.stroke_units=delta/np.maximum(self.stroke_lengths[:,None],1e-9)
        self.angle_limit=angle_limit;self.minimum_body_support=minimum_body_support

    def evaluate(self,line,label,source,target):
        pts=np.asarray(line,dtype=float).reshape(2,2);direction=pts[1]-pts[0];length=float(np.linalg.norm(direction))
        if length<1e-6:return {'accepted':False,'reason':'zero_length'}
        unit=direction/length;radius=max(2.,min(3.,.04*length));extension=max(2.,.15*length)
        endpoint_dist=map_coordinates(self.distance,[pts[:,1],pts[:,0]],order=1,mode='constant',cval=1000)
        endpoint_info=[];reasons=[]
        angles=np.degrees(np.arccos(np.clip(np.abs(self.stroke_units@unit),0,1)))
        for i,atom in enumerate([source,target]):
            aligned=False;best_angle=None;stroke_distance=None
            offset=pts[i]-self.strokes[:,:2]
            projection=np.sum(offset*self.stroke_units,axis=1)
            distances=np.linalg.norm(offset-projection[:,None]*self.stroke_units,axis=1)
            eligible=(self.stroke_lengths>=max(3.,.15*length))&(projection>=-extension)&(projection<=self.stroke_lengths+extension)&(distances<=radius)
            indices=np.flatnonzero(eligible)
            if len(indices):
                best=int(indices[np.argmin(angles[indices])]);best_angle=float(angles[best]);stroke_distance=float(distances[best]);aligned=best_angle<=self.angle_limit
            implicit=atom.get('label_name')=='C'
            x0,y0,x1,y1=atom['bbox_xyxy'];x,y=pts[i]
            outside=not (x0-1<=x<=x1+1 and y0-1<=y<=y1+1)
            supported=not implicit or not outside or endpoint_dist[i]<=radius or aligned
            endpoint_info.append({'ink_distance_px':float(endpoint_dist[i]),'implicit_carbon':implicit,
                                  'outside_atom_box_with_1px_margin':bool(outside),
                                  'aligned_stroke':aligned,'minimum_stroke_angle_deg':best_angle,
                                  'stroke_distance_px':stroke_distance,'supported':bool(supported)})
            if not supported:reasons.append('carbon_endpoint_off_ink_and_unaligned')
        # Sample the interior, excluding explicit glyph boxes. The rule applies
        # to single bonds only; double centers and dash gaps can be white.
        sample=pts[0]+np.linspace(.15,.85,41)[:,None]*direction
        keep=np.ones(len(sample),dtype=bool)
        for atom in [source,target]:
            if atom.get('label_name')=='C':continue
            x0,y0,x1,y1=atom['bbox_xyxy']
            keep&=~((sample[:,0]>=x0-1)&(sample[:,0]<=x1+1)&(sample[:,1]>=y0-1)&(sample[:,1]<=y1+1))
        distances=map_coordinates(self.distance,[sample[:,1],sample[:,0]],order=1,mode='constant',cval=1000)
        support=float(np.mean(distances[keep]<=radius)) if keep.any() else None
        if self.minimum_body_support is not None and label==1 and keep.sum()>=8 and support<self.minimum_body_support:reasons.append('single_bond_interior_without_ink')
        return {'accepted':not reasons,'reason':','.join(reasons),'endpoints':endpoint_info,
                'body_support':support,'body_samples':int(keep.sum()),'ink_radius_px':radius,'angle_limit_deg':self.angle_limit}
