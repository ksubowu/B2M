import unittest
import torch
from .scored_bond_model import scored_targets
from .line_neighborhood_loss import line_neighborhood_terms


class LineNeighborhoodTests(unittest.TestCase):
    def example(self,reverse=False):
        target=scored_targets([4],[[[28,29],[28,31],[36,31],[36,29]]],[[[20,30],[44,30]]],size=16)
        truth=torch.tensor([[20.,30.],[44.,30.]])
        if reverse:truth=truth.flip(0)
        yy,xx=torch.meshgrid(torch.arange(16),torch.arange(16),indexing='ij')
        center=torch.stack((xx,yy),0)*4
        ends=(truth[:,:,None,None]-center[None])/32
        head=dict(endpoints=ends.reshape(1,4,16,16).clone().requires_grad_(),offset=torch.zeros(1,2,16,16,requires_grad=True))
        return target,head

    def test_correct_global_endpoints_at_all_cells_have_zero_loss(self):
        target,head=self.example();terms=line_neighborhood_terms(head,[target])
        for loss in terms.values():self.assertLess(abs(float(loss.detach())),1e-6)

    def test_endpoint_order_is_not_exchange_invariant(self):
        target,head=self.example(reverse=True);terms=line_neighborhood_terms(head,[target])
        self.assertGreater(float(terms['relative'].detach()),0.);self.assertGreater(float(terms['direction'].detach()),1.9)
        sum(terms.values()).backward();self.assertGreater(float(head['endpoints'].grad.abs().sum()),0.)

    def test_neighbor_error_is_supervised_without_changing_center_loss(self):
        target,head=self.example();_,y,x,*_=target[1][0]
        with torch.no_grad():head['endpoints'][0,1,y,x+1]+=.25
        terms=line_neighborhood_terms(head,[target]);self.assertEqual(float(terms['relative'].detach()),0.)
        self.assertGreater(float(terms['neighborhood'].detach()),0.)


if __name__=='__main__':unittest.main()
