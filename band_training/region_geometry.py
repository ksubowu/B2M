"""Geometry-only band region helper used by target invariants."""
import numpy as np


def region_quad(ink, category):
    """Ordered tapered region from visible ink only: no atom boxes or indices."""
    ys, xs = np.nonzero(ink)
    pts = np.column_stack((xs,ys)).astype(float)
    center=pts.mean(0)
    _,vectors=np.linalg.eigh(np.cov((pts-center).T))
    axis=vectors[:,-1]
    if axis[0]<0 or (abs(axis[0])<1e-7 and axis[1]<0): axis=-axis
    normal=np.array([-axis[1],axis[0]])
    t=(pts-center)@axis; u=(pts-center)@normal
    lo,hi=t.min(),t.max(); length=hi-lo
    widths=[]
    for a,b in [(lo,lo+.3*length),(hi-.3*length,hi)]:
        us=u[(t>=a)&(t<=b)]
        widths.append(max(1.,float(np.max(np.abs(us)))+.5))
    # Wedge/dash orientation is measured from widening ink, not atom identities.
    if category in (5,6) and widths[0]>widths[1]:
        axis=-axis;normal=-normal;t=-t;u=-u;lo,hi=-hi,-lo;widths=widths[::-1]
    if category not in (5,6): widths=[max(widths)]*2
    a=center+lo*axis;b=center+hi*axis
    quad=np.stack((a+widths[0]*normal,a-widths[0]*normal,
                   b-widths[1]*normal,b+widths[1]*normal))
    return quad.tolist()
