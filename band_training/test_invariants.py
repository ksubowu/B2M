"""Checks for ordered regions, decoupled targets and geometric assembly."""
import unittest
import numpy as np
import torch
from .model import BandDecoder,band_loss,targets_for,decode
from .geometry import link,edge_key,polygon_iou,suppress,AtomCenterIndex,link_many
from .region_geometry import region_quad


class Invariants(unittest.TestCase):
    def test_crowded_anchors_preserve_exact_regions(self):
        quads=[[[30,29],[30,31],[34,31],[34,29]],
               [[30.5,29],[30.5,31],[34.5,31],[34.5,29]],
               [[31,29],[31,31],[35,31],[35,29]]]
        _,entries=targets_for([0,0,0],quads,'cpu',size=16,resolve_collisions=True)
        self.assertEqual(len(entries),3)
        for entry,expected in zip(entries,quads):
            label,y,x,off,geom=entry
            actual=geom.reshape(4,2)*32+(torch.tensor([x,y])+off)*4
            self.assertTrue(torch.allclose(actual,torch.tensor(expected).float(),atol=1e-5))
        cells=[(e[2],e[1]) for e in entries]
        self.assertEqual(len(set(cells)),3)
        for i,a in enumerate(cells):
            for b in cells[i+1:]:self.assertGreater(max(abs(a[0]-b[0]),abs(a[1]-b[1])),1)

    def test_letterbox_transforms_boxes_and_quads_together(self):
        from .mixed_data import prepare_annotation
        ann=dict(width=100,height=200,
                 atoms=[dict(id=0,symbol='C',formal_charge=0,center=[15,30],bbox=[10,20,20,40]),
                        dict(id=1,symbol='O',formal_charge=-1,center=[65,130],bbox=[60,120,70,140])],
                 charges=[dict(atom_id=1,value=-1,bbox=[70,115,75,120])],
                 bonds=[dict(id=0,source=0,target=1)],bands=[dict(bond_id=0,category_id=1)])
        q=[[[20,40],[22,39],[60,120],[58,121]]]
        metric,box,labels,quads=prepare_annotation(ann,dict(labels=[0],quads=q))
        self.assertTrue(np.allclose(metric['atoms'][0]['bbox'],[192,64,32,64]))
        self.assertTrue(np.allclose(quads,np.asarray(q)*3.2+[160,0]))
        self.assertEqual(box['labels'].tolist(),[1,2,15])

    def test_decode_more_than_160_without_truncation(self):
        out={'heat':torch.full((1,6,64,64),-20.),'offset':torch.zeros(1,2,64,64),
             'quad':torch.zeros(1,48,64,64)}
        for y in range(2,62,3):
            for x in range(2,62,3):out['heat'][0,0,y,x]=5
        self.assertEqual(len(decode(out,minimum=.5)),400)
        self.assertEqual(len(decode(out,minimum=.5,topk=0)),400)
        self.assertEqual(len(decode(out,minimum=.5,topk=160)),160)
        self.assertEqual(len(decode(out,minimum=.999)),0)

    def test_kdtree_matches_brute_force_nearest_centers(self):
        rng=np.random.default_rng(19)
        boxes=np.column_stack((rng.uniform(0,100,(20,2)),rng.uniform(2,15,(20,2))))
        quads=rng.uniform(0,100,(30,4,2))
        centers=boxes[:,:2]+boxes[:,2:]/2
        expected=[]
        for q in quads:
            ends=np.stack((q[:2].mean(0),q[2:].mean(0)))
            indices=np.linalg.norm(ends[:,None]-centers[None],axis=2).argmin(1)
            expected.append(tuple(map(int,indices)) if indices[0]!=indices[1] and np.linalg.norm(ends[1]-ends[0])>=3 else None)
        self.assertEqual(AtomCenterIndex(boxes).link_many(quads),expected)

    def test_kdtree_preserves_order_and_rejects_self_edge(self):
        boxes=[[8,8,4,4],[88,8,4,4]]
        q=[[15,9],[15,11],[85,11],[85,9]]
        self.assertEqual(link_many([q],boxes,'kdtree'),[(0,1)])
        self.assertEqual(link_many([np.asarray(q)[[3,2,1,0]]],boxes,'kdtree'),[(1,0)])
        same=[[11,9],[11,11],[16,11],[16,9]]
        self.assertEqual(link_many([same],boxes,'kdtree'),[None])
        self.assertEqual(link_many([],boxes,'kdtree'),[])
        self.assertEqual(link_many([q],[],'kdtree'),[None])

    def test_independent_band_gradient(self):
        torch.set_num_threads(2)
        model=BandDecoder()
        feature=[torch.randn(1,256,8,8)]
        gray=torch.randn(1,1,64,64)
        # Target carries only type and geometry. No boxes, indices or pair list.
        target=targets_for([0],[[[18,29],[18,31],[46,31],[46,29]]],'cpu',size=16)
        out=model(feature,gray);loss=band_loss(out,[target])['total'];loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.quad.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.heat.weight.grad.abs().sum()),0)

    def test_order_is_not_swapped(self):
        target=targets_for([4],[[[18,29],[18,31],[46,35],[46,25]]],'cpu',size=16)
        label,y,x,offset,quad=target[1][0]
        out={'heat':torch.zeros(1,6,16,16),'offset':torch.zeros(1,2,16,16),'quad':torch.zeros(1,48,16,16)}
        out['quad'][0,label*8:label*8+8,y,x]=quad
        correct=band_loss(out,[target])['region']
        out['quad'][0,label*8:label*8+8,y,x]=quad.reshape(4,2)[[3,2,1,0]].flatten()
        reversed_loss=band_loss(out,[target])['region']
        self.assertAlmostEqual(float(correct),0)
        self.assertGreater(float(reversed_loss),.01)

    def test_gap_and_middle_atom(self):
        quad=[[40,48],[40,52],[80,52],[80,48]]
        boxes=[[14,44,12,12],[94,44,12,12],[54,44,12,12]]
        self.assertEqual(link(quad,boxes),(0,1))
        self.assertNotEqual(edge_key(0,1,4),edge_key(1,0,4))
        self.assertEqual(edge_key(0,1,0),edge_key(1,0,0))

    def test_duplicate_region(self):
        q=[[10,10],[10,15],[50,15],[50,10]]
        self.assertAlmostEqual(polygon_iou(q,q),1)
        self.assertEqual(len(suppress([dict(quad=q,label=0,score=.9),dict(quad=q,label=1,score=.8)])),1)

    def test_region_derived_from_pixels(self):
        ink=np.zeros((64,64),bool);ink[29:32,12:50]=True
        q=np.asarray(region_quad(ink,1))
        self.assertLess(q[:2,0].mean(),q[2:,0].mean())
        self.assertTrue(np.isfinite(q).all())

if __name__=='__main__':unittest.main()
