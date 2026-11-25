import unittest
import numpy as np
import torch
from .scored_bond_model import ScoredBondDecoder,scored_targets,scored_bond_loss,decode_scored_bonds
from .scored_bond_postprocess import assemble_scored_bonds,group_lines,refine_peak_lines


def output(size=8):
    return dict(line=dict(score_logits=torch.full((1,1,size,size),-30.),offset=torch.zeros(1,2,size,size),endpoints=torch.zeros(1,4,size,size)),
        band=dict(score_logits=torch.full((1,1,size,size),-30.),offset=torch.zeros(1,2,size,size),type_logits=torch.zeros(1,6,size,size),quad=torch.zeros(1,48,size,size)))


class ScoredBondTests(unittest.TestCase):
    def test_peak_refinement_keeps_count_presence_and_band_query(self):
        peak=dict(query_id=162,line_score=.9,line=[[0,7],[20,7]],is_local_peak=True)
        neighbor=dict(query_id=163,line_score=.8,line=[[0,0],[20,0]],is_local_peak=False)
        decoded=dict(lines=[peak,neighbor],bands=[])
        refined=refine_peak_lines([[-1,-1,2,2],[19,-1,2,2]],decoded)
        self.assertEqual(len(refined['lines']),1);r=refined['lines'][0]
        self.assertEqual(r['query_id'],162);self.assertEqual(r['geometry_query_id'],163)
        self.assertEqual(r['line_score'],.9);self.assertEqual(r['line'],neighbor['line'])
        self.assertEqual(peak['line'],[[0,7],[20,7]])

    def test_optional_stereo_band_direction_preserves_input_and_positions(self):
        lines=[dict(query_id=1,line_score=.9,line=[[20,0],[0,0]])]
        band=dict(query_id=1,band_score=.9,type_score=1.,score=.9,label=4,quad=[[0,1],[0,-1],[20,-4],[20,4]])
        decoded=dict(lines=lines,bands=[band]);boxes=[[-1,-1,2,2],[19,-1,2,2]]
        original=assemble_scored_bonds(boxes,decoded);fixed=assemble_scored_bonds(boxes,decoded,stereo_from_band=True)
        self.assertEqual((original['bonds'][0]['begin'],original['bonds'][0]['end']),(1,0))
        self.assertEqual((fixed['bonds'][0]['begin'],fixed['bonds'][0]['end']),(0,1))
        self.assertEqual(lines[0]['line'],[[20,0],[0,0]])
        self.assertEqual(fixed['instances'][0]['line'],[[0.,0.],[20.,0.]])

    def test_diagnostic_decode_can_keep_nonpeak_lines(self):
        out=output();out['line']['score_logits'][0,0,2,3]=3;out['line']['score_logits'][0,0,2,4]=2
        a=decode_scored_bonds(out);b=decode_scored_bonds(out,apply_line_peaks=False)
        self.assertEqual(len(a['lines']),1);self.assertEqual(len(b['lines']),2)
        self.assertEqual(sum(r['is_local_peak'] for r in b['lines']),1)

    def test_exclusive_assignment_keeps_untyped_line_anchors(self):
        boxes=[[-1,-1,2,2],[9,-1,2,2],[19,-1,2,2]]
        lines=[dict(query_id=1,line_score=.9,line=[[0,0],[10,0]]),dict(query_id=2,line_score=.8,line=[[10,0],[20,0]])]
        band=dict(query_id=1,band_score=.9,type_score=1.,score=.9,label=4,quad=[[0,1],[0,-1],[10,-1],[10,1]])
        g=assemble_scored_bonds(boxes,dict(lines=lines,bands=[band]),line_association='exclusive_box_noisy',endpoint_cluster_radius=2.)
        self.assertEqual(len(g['untyped_lines']),1);self.assertEqual(len(g['line_graph']),2)
        self.assertEqual([(e['begin'],e['end'],e['label']) for e in g['bonds']],[(0,1,4)])

    def test_line_survives_without_band_score(self):
        out=output();out['line']['score_logits'][0,0,2,3]=10.;out['line']['endpoints'][0,:,2,3]=torch.tensor([-1.,0.,1.,0.])
        decoded=decode_scored_bonds(out)
        self.assertEqual(len(decoded['lines']),1);self.assertEqual(decoded['bands'],[])
        graph=assemble_scored_bonds([[-21,7,2,2],[43,7,2,2]],decoded)
        self.assertEqual(len(graph['untyped_lines']),1)
        self.assertEqual(graph['bonds'],[])

    def test_band_survives_without_line_score(self):
        out=output();out['band']['score_logits'][0,0,2,3]=10;out['band']['type_logits'][0,1,2,3]=10
        decoded=decode_scored_bonds(out)
        self.assertEqual(len(decoded['bands']),1);self.assertEqual(decoded['lines'],[])
        self.assertEqual(decoded['bands'][0]['label'],1)

    def test_each_detection_set_uses_only_its_own_threshold(self):
        out=output();out['line']['score_logits'][0,0,2,3]=3
        out['band']['score_logits'][0,0,2,3]=2;out['band']['score_logits'][0,0,2,4]=4
        out['band']['type_logits'][0,1,2,3]=10;out['band']['type_logits'][0,1,2,4]=10
        a=decode_scored_bonds(out,line_threshold=.2,band_threshold=.2)
        b=decode_scored_bonds(out,line_threshold=.99,band_threshold=.2)
        c=decode_scored_bonds(out,line_threshold=.2,band_threshold=.99)
        self.assertEqual([r['query_id'] for r in a['bands']],[r['query_id'] for r in b['bands']])
        self.assertEqual([r['query_id'] for r in a['lines']],[r['query_id'] for r in c['lines']])
        self.assertEqual(len(a['paired_band_probes']),1)
        self.assertEqual(len(b['paired_band_probes']),0)

    def test_one_band_type_per_query_and_separate_scores(self):
        out=output();out['line']['score_logits'][0,0,2,3]=2;out['band']['score_logits'][0,0,2,3]=5
        out['band']['type_logits'][0,1,2,3]=5;out['band']['type_logits'][0,0,2,3]=4
        decoded=decode_scored_bonds(out)
        self.assertEqual(len(decoded['bands']),1)
        self.assertEqual(decoded['bands'][0]['query_id'],decoded['lines'][0]['query_id'])
        self.assertNotEqual(decoded['bands'][0]['band_score'],decoded['lines'][0]['line_score'])

    def test_crowded_targets_preserve_all_types_and_geometry(self):
        quads=[[[28,29],[28,31],[36,31],[36,29]],[[28.1,29],[28.1,31],[36.1,31],[36.1,29]]]
        lines=[[[20,30],[44,30]],[[21,30],[45,30]]]
        heat,entries=scored_targets([0,1],quads,lines,size=16)
        self.assertEqual(int(heat.eq(1).sum()),2)
        cells=[]
        for (_,y,x,off,quad,line),q,l in zip(entries,quads,lines):
            center=(torch.tensor([x,y])+off)*4
            torch.testing.assert_close(quad.reshape(4,2)*32+center,torch.tensor(q).float(),atol=1e-5,rtol=0)
            torch.testing.assert_close(line.reshape(2,2)*32+center,torch.tensor(l).float(),atol=1e-5,rtol=0)
            cells.append((x,y))
        self.assertGreater(max(abs(cells[0][i]-cells[1][i]) for i in [0,1]),1)

    def test_each_score_has_its_own_gradients(self):
        torch.set_num_threads(2);m=ScoredBondDecoder()
        features=[torch.randn(1,256,8,8)];gray=torch.randn(1,1,64,64)
        target=scored_targets([1],[[[18,29],[18,31],[46,31],[46,29]]],[[[12,30],[52,30]]],size=16)
        result=scored_bond_loss(m(features,gray),[target]);result['total'].backward()
        for p in [m.line_head.score.weight,m.band_head.score.weight,m.line_head.endpoints.weight,m.band_head.quad.weight,m.band_head.types.weight]:
            self.assertGreater(float(p.grad.abs().sum()),0)
        self.assertNotEqual(m.line_head.score.weight.data_ptr(),m.band_head.score.weight.data_ptr())

    def test_band_loss_does_not_train_line_score(self):
        m=ScoredBondDecoder();target=scored_targets([0],[[[18,29],[18,31],[46,31],[46,29]]],[[[12,30],[52,30]]],size=16)
        losses=scored_bond_loss(m([torch.randn(1,256,8,8)],torch.randn(1,1,64,64)),[target])
        losses['band_total'].backward()
        self.assertIsNone(m.line_head.score.weight.grad)
        self.assertIsNotNone(m.band_head.score.weight.grad)

    def test_geometry_and_band_score_can_choose_different_queries(self):
        lines=[dict(query_id=1,line_score=.9,line=[[0,0],[20,0]]),dict(query_id=2,line_score=.7,line=[[.2,0],[20.2,0]])]
        bands=[dict(query_id=1,band_score=.4,type_score=.9,score=.4,label=0,quad=[[1,1],[1,-1],[19,-1],[19,1]]),
               dict(query_id=2,band_score=.95,type_score=.99,score=.95,label=1,quad=[[1,2],[1,-2],[19,-2],[19,2]])]
        graph=assemble_scored_bonds([[-1,-1,2,2],[19,-1,2,2]],dict(lines=lines,bands=bands))
        self.assertEqual(len(graph['bonds']),1);self.assertEqual(graph['bonds'][0]['label'],1)
        self.assertEqual(graph['instances'][0]['line_query'],1);self.assertEqual(graph['instances'][0]['band_query'],2)
        np.testing.assert_array_equal(graph['instances'][0]['line'],lines[0]['line'])

    def test_ordered_line_loss_does_not_swap_endpoints(self):
        target=scored_targets([4],[[[18,29],[18,31],[46,35],[46,25]]],[[[12,30],[52,30]]],size=16)
        out=output(16);_,y,x,off,q,line=target[1][0]
        out['line']['endpoints'][0,:,y,x]=line
        self.assertEqual(float(scored_bond_loss(out,[target])['line_endpoints']),0)
        out['line']['endpoints'][0,:,y,x]=line.reshape(2,2).flip(0).flatten()
        self.assertGreater(float(scored_bond_loss(out,[target])['line_endpoints']),0)

    def test_same_atom_pair_combines_best_line_and_best_band(self):
        lines=[dict(query_id=1,line_score=.95,line=[[0,0],[20,0]]),dict(query_id=2,line_score=.7,line=[[3,0],[17,0]])]
        band=dict(query_id=2,band_score=.9,type_score=.99,score=.9,label=1,quad=[[3,2],[3,-2],[17,-2],[17,2]])
        graph=assemble_scored_bonds([[-1,-1,2,2],[19,-1,2,2]],dict(lines=lines,bands=[band]))
        self.assertEqual(len(graph['line_graph']),1);self.assertEqual(len(graph['bonds']),1)
        self.assertEqual(len(graph['untyped_lines']),0)
        self.assertEqual(graph['instances'][0]['line_query'],1);self.assertEqual(graph['instances'][0]['band_query'],2)

if __name__=='__main__':unittest.main()
