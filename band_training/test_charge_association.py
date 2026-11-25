import unittest
from .charge_association import assign_charges

class ChargeAssociationTests(unittest.TestCase):
    def test_negative_oxygen_prefers_single_bond_over_nearer_double_bond(self):
        atoms=[dict(symbol='P',bbox=[5,20,4,4]),dict(symbol='O',bbox=[10,0,8,8]),dict(symbol='O',bbox=[0,0,8,8])]
        bonds=[dict(begin=0,end=1,type='double'),dict(begin=0,end=2,type='single')]
        charges=[dict(value=-1,bbox=[8.5,0,2,2])]
        self.assertEqual(assign_charges(atoms,charges,bonds,'legacy')[0]['atom_index'],1)
        self.assertEqual(assign_charges(atoms,charges,bonds)[0]['atom_index'],2)
    def test_opposite_charges_do_not_cancel_on_same_atom(self):
        atoms=[dict(symbol='N',bbox=[0,10,10,10]),dict(symbol='O',bbox=[8,0,10,10]),dict(symbol='O',bbox=[2,25,8,8]),dict(symbol='C',bbox=[-10,10,5,5])]
        bonds=[dict(begin=0,end=1,type='single'),dict(begin=0,end=2,type='double'),dict(begin=0,end=3,type='single')]
        charges=[dict(value=1,bbox=[9,9,2,2]),dict(value=-1,bbox=[16,0,2,2])]
        self.assertEqual([v['atom_index'] for v in assign_charges(atoms,charges,bonds)],[0,1])
    def test_gt_charge_attributes_never_used(self):
        atoms=[dict(symbol='O',bbox=[0,0,6,6],formal_charge=100,atom_id=999),dict(symbol='C',bbox=[30,0,6,6],formal_charge=-100)]
        charges=[dict(value=-1,bbox=[6,0,2,2],atom_id=999)]
        first=assign_charges(atoms,charges)
        for a in atoms:a['formal_charge']=0;a['atom_id']=-1
        charges[0]['atom_id']=-42
        self.assertEqual(first,assign_charges(atoms,charges))
    def test_duplicate_or_far_charges_can_remain_unassigned(self):
        atoms=[dict(symbol='other',bbox=[0,0,4,4]),dict(symbol='other',bbox=[200,0,4,4])]
        charges=[dict(value=1,bbox=[4,0,1,1]),dict(value=-1,bbox=[4,0,1,1])]
        owners=[v['atom_index'] for v in assign_charges(atoms,charges)]
        self.assertIn(None,owners);self.assertEqual(owners.count(0),1)

    def test_nonstandard_charge_is_not_hard_rejected_when_geometry_clear(self):
        atoms=[dict(symbol='N',bbox=[0,0,8,8])]+[dict(symbol='C',bbox=[40+i*20,20,8,8]) for i in range(3)]
        bonds=[dict(begin=0,end=i,type='single') for i in range(1,4)]
        charges=[dict(value=-1,bbox=[8,0,2,2])]
        self.assertEqual(assign_charges(atoms,charges,bonds)[0]['atom_index'],0)

if __name__=='__main__':unittest.main()
