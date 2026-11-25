import unittest
from .endpoint_assignment import associate_unique_endpoints

class EndpointAssignmentTests(unittest.TestCase):
    def test_shared_line_endpoints_are_unique_atoms(self):
        boxes=[[-1,-1,2,2],[9,-1,2,2],[19,-1,2,2]];lines=[[[0,0],[10,0]],[[10,0],[20,0]]]
        pairs,a=associate_unique_endpoints(boxes,lines)
        self.assertEqual(a['unique_endpoints'],3);self.assertEqual(pairs,[(0,1),(1,2)])
    def test_reverse_nearest_collision_is_not_silently_overwritten(self):
        boxes=[[0,-5,200,10],[49,-1,2,2]];lines=[[[0,0],[50,0]]]
        pairs,a=associate_unique_endpoints(boxes,lines)
        self.assertEqual(pairs,[None]);self.assertEqual(a['raw_nearest_collision_endpoints'],1)
        pairs,a=associate_unique_endpoints(boxes,lines,'exclusive_box')
        self.assertEqual(pairs,[(0,1)]);self.assertEqual(a['locked_exact_centers'],1)
    def test_isolated_atom_need_not_have_a_line_endpoint(self):
        boxes=[[-1,-1,2,2],[9,-1,2,2],[99,-1,2,2]];lines=[[[0,0],[10,0]]]
        pairs,a=associate_unique_endpoints(boxes,lines,'exclusive_box')
        self.assertEqual(pairs,[(0,1)]);self.assertFalse(a['endpoint_count_matches_atoms'])
        self.assertEqual(a['boxes_without_endpoint'],1)
    def test_endpoint_order_and_box_permutation(self):
        boxes=[[9,-1,2,2],[-1,-1,2,2]];lines=[[[0,0],[10,0]]]
        self.assertEqual(associate_unique_endpoints(boxes,lines,'exclusive_box')[0],[(1,0)])

if __name__=='__main__':unittest.main()
