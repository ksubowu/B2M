import unittest
from .prepare_v6_prior_control import select,features


class PriorSelectionTests(unittest.TestCase):
    def test_unique_selection_keeps_original_and_obeys_training_priors(self):
        records=[];metadata={};ids=set()
        for i in range(10):
            sid='base'+str(i);records.append(dict(sample_id=sid));ids.add(sid)
            metadata[sid]=dict(atom_classes={str(i+8):1} if i<5 else {'1':10},bond_classes={'0':10,'3':int(i==0)})
        for i in range(200):
            sid='new'+str(i);records.append(dict(sample_id=sid))
            metadata[sid]=dict(atom_classes={str(k):1 for k in [8,9,10,11,12]} if i<150 else {'1':10},bond_classes={'0':10,'3':int(i<150)})
        rows,stats=select(records,metadata,20,ids,seed=42)
        self.assertEqual(len(rows),20);self.assertEqual(len({r['sample_id'] for r in rows}),20)
        self.assertTrue(ids<={r['sample_id'] for r in rows})
        self.assertTrue(all(stats['selected_feature_images'][k]<=cap for k,cap in stats['caps'].items()))
        self.assertEqual(rows,select(records,metadata,20,ids,seed=42)[0])


if __name__=='__main__':unittest.main()
