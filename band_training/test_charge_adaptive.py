import unittest
from .charge_adaptive import assign_adaptive
from .charge_association import assign_charges

class AdaptiveChargeTests(unittest.TestCase):
    def test_plausible_protonated_nitrogen_is_not_moved_to_carbon(self):
        atoms=[dict(symbol='N',bbox=[0,0,8,8]),dict(symbol='C',bbox=[10,0,8,8])]+[dict(symbol='C',bbox=[40+i*20,20,8,8]) for i in range(3)]
        bonds=[dict(begin=0,end=2,type='single'),dict(begin=1,end=2,type='single'),dict(begin=1,end=3,type='single'),dict(begin=1,end=4,type='single')]
        charges=[dict(value=1,bbox=[6,0,4,4])];lines=[[[4,4],[44,24]]]
        baseline=assign_charges(atoms,charges,bonds)
        self.assertEqual(baseline[0]['atom_index'],0)
        self.assertEqual(assign_adaptive(atoms,charges,bonds,lines,preserve_positive_hetero=True)[0]['atom_index'],0)
    def test_scale_aware_search_reaches_quaternary_nitrogen(self):
        atoms=[dict(symbol='C',bbox=[0,0,8,8]),dict(symbol='N',bbox=[15,0,8,8])]+[dict(symbol='C',bbox=[50+i*20,20,8,8]) for i in range(5)]
        bonds=[dict(begin=0,end=2,type='double'),dict(begin=0,end=3,type='single'),dict(begin=0,end=4,type='single')]+[dict(begin=1,end=i,type='single') for i in range(2,6)]
        charges=[dict(value=1,bbox=[4,0,6,6])];lines=[[[4,4],[19,4]]]
        self.assertEqual(assign_charges(atoms,charges,bonds,method='safe_adaptive',lines=lines)[0]['atom_index'],1)
    def test_public_api_requires_line_scale_input(self):
        with self.assertRaises(ValueError):assign_charges([],[],method='safe_adaptive')
        self.assertEqual(assign_charges([],[],method='safe_adaptive',lines=[]),[])

if __name__=='__main__':unittest.main()
