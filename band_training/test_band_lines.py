import unittest
import numpy as np
from .geometry import assemble_band_lines,LineEndpointIndex

class BandLineTests(unittest.TestCase):
    def test_short_complete_line_is_not_discarded(self):
        boxes=[[0,0,1,1],[1,0,1,1]]
        self.assertEqual(LineEndpointIndex(boxes).link_many([[[.5,.5],[1.5,.5]]]),[(0,1)])
    def test_class_from_band_connection_from_line(self):
        boxes=[[-1,-1,2,2],[99,-1,2,2],[199,-1,2,2]]
        # Visible ink ends would be nearest to a different atom; line wins.
        item=dict(label=1,quad=[[180,2],[180,-2],[200,-2],[200,2]],line=[[0,0],[100,0]])
        graph=assemble_band_lines(boxes,[item]);self.assertEqual(graph['bonds'][0]['type'],'double')
        self.assertEqual((graph['bonds'][0]['begin'],graph['bonds'][0]['end']),(0,1))
    def test_stereo_direction_is_preserved_and_atom_order_is_not_assumed(self):
        boxes=[[99,-1,2,2],[-1,-1,2,2]]
        item=dict(label=4,quad=[[3,1],[3,-1],[97,-3],[97,3]],line=[[0,0],[100,0]])
        graph=assemble_band_lines(boxes,[item]);self.assertEqual((graph['bonds'][0]['begin'],graph['bonds'][0]['end']),(1,0))
    def test_invalid_and_self_connections_are_rejected(self):
        index=LineEndpointIndex([[0,0,4,4],[40,0,4,4]])
        self.assertEqual(index.link_many([[[2,2],[2,2]],[[np.nan,0],[42,2]]]),[None,None])
        with self.assertRaises(ValueError):assemble_band_lines([[0,0,4,4]],[dict(label=0,quad=[[0,0]]*4)])

if __name__=='__main__':unittest.main()
