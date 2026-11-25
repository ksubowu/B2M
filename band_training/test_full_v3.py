import unittest
import numpy as np
import torch
from .model import targets_for,decode
from .band_line_model import paired_bond_loss,BandLineDecoder
from .endpoint_assignment import associate_unique_endpoints
from .full_v3_data import scaled_lines,orient_targets


class FullV3Tests(unittest.TestCase):
    def test_canonical_order_is_fixed_and_stereo_direction_is_kept(self):
        quad=[[10,1],[10,-1],[0,-1],[0,1]]
        q,l=orient_targets([0,4],[quad,quad],[[[10,0],[0,0]],[[10,0],[0,0]]])
        self.assertEqual(l[0],[[0,0],[10,0]])
        self.assertEqual(l[1],[[10,0],[0,0]])
        np.testing.assert_allclose(np.mean(q[0][:2],0),[0,0])

    def test_ordered_line_loss_does_not_exchange_ends(self):
        quad=[[8,12],[8,8],[32,8],[32,12]];line=[[[4,10],[36,10]]]
        target=targets_for([0],[quad],'cpu')
        out=dict(heat=torch.zeros(1,6,160,160),offset=torch.zeros(1,2,160,160),
                 quad=torch.zeros(1,48,160,160),line=torch.zeros(1,24,160,160))
        _,y,x,offset,_=target[1][0];center=(torch.tensor([x,y])+offset)*4
        out['line'][0,:4,y,x]=((torch.tensor(line[0])-center)/32).flatten()
        self.assertEqual(float(paired_bond_loss(out,[target],[line])['line']),0.)
        self.assertGreater(float(paired_bond_loss(out,[target],[[line[0][::-1]]])['line']),0.)

    def test_decode_pairs_same_class_and_peak(self):
        out=dict(heat=torch.full((1,6,4,4),-20.),offset=torch.zeros(1,2,4,4),
                 quad=torch.zeros(1,48,4,4),line=torch.zeros(1,24,4,4))
        out['heat'][0,4,2,1]=10
        out['line'][0,16:20,2,1]=torch.tensor([-1.,0.,1.,0.])
        p=decode(out,minimum=.5)[0]
        self.assertEqual(p['label'],4)
        np.testing.assert_allclose(p['line'],[[-28,8],[36,8]])

    def test_noise_clusters_shared_atom_without_merging_bond_ends(self):
        boxes=[[-1,-1,2,2],[9,-1,2,2],[19,-1,2,2]]
        lines=[[[0,0],[9.8,.1]],[[10.2,-.1],[20,0]]]
        pairs,a=associate_unique_endpoints(boxes,lines,'exclusive_box',cluster_radius=1.)
        self.assertEqual(pairs,[(0,1),(1,2)]);self.assertEqual(a['unique_endpoints'],3)
        pairs,a=associate_unique_endpoints([[-1,-1,2,2],[0,-1,2,2]],[[[0,0],[1,0]]],'exclusive_box',cluster_radius=2.)
        self.assertEqual(a['unique_endpoints'],2);self.assertEqual(pairs,[(0,1)])

    def test_complete_diameter_stops_chaining(self):
        lines=[[[0,0],[20,0]],[[.8,0],[30,0]],[[1.6,0],[40,0]]]
        _,a=associate_unique_endpoints([],lines,'exclusive_box',cluster_radius=1.)
        self.assertEqual(a['unique_endpoints'],5)

    def test_noisy_surplus_endpoints_attach_to_established_anchor(self):
        boxes=[[-1,-1,2,2],[19,-1,2,2],[39,-1,2,2]]
        lines=[[[0,0],[18,0]],[[22,0],[40,0]]]
        pairs,a=associate_unique_endpoints(boxes,lines,'exclusive_box_noisy',cluster_radius=1.)
        self.assertEqual(pairs,[(0,1),(1,2)])
        self.assertEqual(a['recovered_split_endpoint_clusters'],1)

    def test_postprocessor_selection_uses_predicted_graph_validation(self):
        from .train_full_v3 import choose_association
        metrics=dict(full_graph_exact=.1,bond_graph=dict(f1=.8),forward_full_graph_exact=.3,
                     forward_baseline=dict(f1=.9),charge_f1=.95)
        method,score=choose_association(metrics,'exclusive_box_noisy')
        self.assertEqual(method,'forward')

    def test_rounded_letterbox_line_alignment(self):
        ann=dict(width=1000,height=333)
        target=dict(labels=[4],lines=[[[0,0],[1000,333]]])
        lines=scaled_lines(ann,dict(labels=[4],quads=[[]]),target)
        np.testing.assert_allclose(lines,[[[0,213],[640,426]]])
        with self.assertRaises(ValueError):scaled_lines(ann,dict(labels=[0],quads=[[]]),target)

if __name__=='__main__':unittest.main()
