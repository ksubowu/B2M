import unittest
import torch
from .charge_loss import charge_terms


class ChargeLossTests(unittest.TestCase):
    def test_positive_class_gradient_when_box_iou_zero(self):
        logits=torch.zeros(1,2,18,requires_grad=True)
        boxes=torch.tensor([[[.1,.1,.1,.1],[.3,.3,.1,.1]]],requires_grad=True)
        targets=[dict(labels=torch.tensor([16]),boxes=torch.tensor([[.8,.8,.02,.02]]))]
        terms=charge_terms(dict(pred_logits=logits,pred_boxes=boxes),targets,[(torch.tensor([0]),torch.tensor([0]))])
        sum(terms.values()).backward()
        self.assertLess(float(logits.grad[0,0,16]),0)
        self.assertGreater(float(logits.grad[0,1,16]),0)
        self.assertTrue(torch.isfinite(boxes.grad).all())
        self.assertGreater(float(boxes.grad.abs().sum()),0)

    def test_correct_box_has_zero_localization_loss(self):
        b=torch.tensor([[[.5,.5,.015,.003]]],requires_grad=True)
        terms=charge_terms(dict(pred_logits=torch.zeros(1,1,18),pred_boxes=b),
             [dict(labels=torch.tensor([15]),boxes=b[0].detach())],[(torch.tensor([0]),torch.tensor([0]))])
        self.assertAlmostEqual(float(terms['relative_box'].detach()),0,places=6)
        self.assertAlmostEqual(float(terms['giou'].detach()),0,places=6)

    def test_no_charge_still_penalizes_false_positives(self):
        logits=torch.zeros(1,1,18,requires_grad=True);b=torch.tensor([[[.5,.5,.02,.02]]],requires_grad=True)
        terms=charge_terms(dict(pred_logits=logits,pred_boxes=b),
            [dict(labels=torch.tensor([2]),boxes=b[0].detach())],[(torch.tensor([0]),torch.tensor([0]))])
        sum(terms.values()).backward()
        self.assertTrue((logits.grad[:,:,13:]>0).all())
        self.assertTrue((logits.grad[:,:,:13]==0).all())

    def test_hard_negative_excludes_real_positive(self):
        logits=torch.zeros(1,2,18)
        logits[0,0,16]=8;logits[0,1,15]=8;logits.requires_grad_()
        boxes=torch.tensor([[[.5,.5,.02,.02],[.2,.2,.02,.02]]])
        terms=charge_terms(dict(pred_logits=logits,pred_boxes=boxes),
            [dict(labels=torch.tensor([16]),boxes=boxes[0,:1])],[(torch.tensor([0]),torch.tensor([0]))])
        terms['hard_negative'].backward()
        self.assertEqual(float(logits.grad[0,0,16]),0)
        self.assertGreater(float(logits.grad[0,1,15]),0)

if __name__=='__main__':unittest.main()
