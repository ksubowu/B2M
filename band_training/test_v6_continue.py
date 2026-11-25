import unittest
from .train_v6_continue import select_smoke,key,acceptable,extended_lr_factor,configure_box_training


class V6ContinuationTests(unittest.TestCase):
    def test_smoke_is_unique_and_retains_charged_and_rare_examples(self):
        records=[dict(sample_id=str(i)) for i in range(200)];metadata=[]
        for i in range(200):
            metadata.append(dict(sample_id=str(i),atoms=i%30+1,charges=int(i<12),
                atom_classes={'1':10,**({'12':1} if i==199 else {})},charge_classes={'16':1} if i<12 else {},bond_classes={'4':1} if i%7==0 else {'0':10}))
        selected=select_smoke(records,metadata,64,42,include_all_charged=True)
        self.assertEqual(len(selected),64);self.assertEqual(len(set(selected)),64)
        self.assertIn(199,selected);self.assertTrue(set(range(12))<=set(selected))
        self.assertEqual(selected,select_smoke(records,metadata,64,42,include_all_charged=True))

    def test_extension_preserves_lr_and_decays_over_additional_epochs(self):
        start=17562;end=193182;factor=.775
        self.assertAlmostEqual(extended_lr_factor(start,start,end,factor),factor)
        self.assertAlmostEqual(extended_lr_factor(end,start,end,factor),factor*.1)
        values=[extended_lr_factor(step,start,end,factor) for step in range(start,end,1000)]
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))

    def test_frozen_box_decoder_disables_gradients_and_training_buffers(self):
        import torch
        from types import SimpleNamespace
        decoder=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4),torch.nn.Dropout(.5))
        base=SimpleNamespace(box_decoder=decoder)
        decoder.train();configure_box_training(base,True)
        self.assertFalse(decoder.training);self.assertTrue(all(not p.requires_grad for p in decoder.parameters()))
        before=decoder[1].running_mean.clone();decoder(torch.ones(8,4))
        self.assertTrue(torch.equal(before,decoder[1].running_mean))
        configure_box_training(base,False);self.assertTrue(all(p.requires_grad for p in decoder.parameters()))

    def test_graph_selection_and_regression_guards(self):
        baseline=dict(atom_f1=.995,charge_f1=.90,typed_bond=dict(f1=.99),graph_ocr_correct=100,
            graph_star_correct=110,counts=dict(full_graph_exact=110),rare_atom_macro_f1=.8)
        gain=dict(baseline,graph_ocr_correct=101)
        self.assertGreater(key(gain),key(baseline));self.assertTrue(acceptable(gain,baseline))
        regression=dict(gain,charge_f1=.80)
        self.assertFalse(acceptable(regression,baseline))


if __name__=='__main__':unittest.main()
