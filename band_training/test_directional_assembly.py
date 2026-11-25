import unittest
import numpy as np
from .geometry import AtomCenterIndex,DirectionalBandIndex


class DirectionalAssemblyTests(unittest.TestCase):
    def test_clipped_visible_bond_does_not_attach_both_ends_to_same_atom(self):
        boxes=[[-4,-4,8,8],[96,-4,8,8]]
        quad=[[[60,2],[60,-2],[100,-2],[100,2]]]
        self.assertEqual(AtomCenterIndex(boxes).link_many(quad),[None])
        self.assertEqual(DirectionalBandIndex(boxes).link_many(quad),[(0,1)])

    def test_long_text_box_anchors_remain_one_original_atom(self):
        boxes=[[-60,1,10,10],[0,0,100,12]]
        quad=[[[-45,8],[-45,4],[-10,4],[-10,8]]]
        self.assertEqual(DirectionalBandIndex(boxes).link_many(quad),[(0,1)])

    def test_degenerate_and_empty_inputs(self):
        index=DirectionalBandIndex([[0,0,4,4],[40,0,4,4]])
        self.assertEqual(index.link_many([]),[])
        self.assertEqual(index.link_many([[[5,5]]*4]),[None])

    def test_adaptive_kdtree_matches_exhaustive_directional_search(self):
        rng=np.random.default_rng(715)
        boxes=np.column_stack((rng.uniform(0,600,(100,2)),rng.uniform(4,50,(100,2))))
        index=DirectionalBandIndex(boxes);quads=[];expected=[]
        for _ in range(30):
            a,b=rng.uniform(0,640,(2,2));axis=(b-a)/np.linalg.norm(b-a);normal=np.array([-axis[1],axis[0]])
            q=np.array([a+normal*2,a-normal*2,b-normal*2,b+normal*2]);quads.append(q);pair=[]
            for end,direction in [(a,-axis),(b,axis)]:
                delta=index.points-end;outward=delta@direction;perp=np.abs(delta[:,0]*direction[1]-delta[:,1]*direction[0])
                score=np.linalg.norm(delta,axis=1)+2*perp+2*np.maximum(0,-outward-2)
                score=np.where((index.points-(a+b)/2)@direction>0,score,np.inf)
                pair.append(int(index.owners[score.argmin()]) if np.isfinite(score.min()) else -1)
            expected.append(tuple(pair) if min(pair)>=0 and pair[0]!=pair[1] else None)
        self.assertEqual(index.link_many(quads),expected)

if __name__=='__main__':unittest.main()
