import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from . import partition_mini_hard as m


class MiniPartitionTests(unittest.TestCase):
    def fixture(self,temp):
        root=Path(temp)/'data';out=Path(temp)/'review';root.mkdir();out.mkdir();records=[]
        for name,split in [('tiny-train','train'),('tiny-val','val'),('regular-train','train'),('regular-val','val')]:
            r=dict(sample_id=name,split=split,group='group-'+name,dataset='test',variant='bw',SMILES='C',source_SMILES='C',has_charge=False,is_color=False)
            for key in m.ASSETS:
                relative='assets/'+name+'/'+key+'.bin';r[key]=relative;p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(relative.encode())
            records.append(r)
        # A shared file must be copied, not removed from normal assets.
        records[0]['band_target']=records[2]['band_target']
        train=[records[0],records[2],dict(records[0],sample_id='tiny-train::repeat_1',asset_sample_id='tiny-train')]
        manifests=dict(data_manifest=records,train=train,unique_train=[records[0],records[2]],val=[records[1],records[3]],
                       source_train=[records[0],records[2]],charge_manifest=[],render_manifest=records)
        for name,rows in manifests.items():m.dump(root/(name+'.json'),rows);m.write_csv(root/(name+'.csv'),rows)
        m.dump(root/'dataset_config.json',dict(charge_balance=dict(enabled=True,charged_fraction=.25)))
        m.dump(root/'validation_report.json',dict(status='passed'))
        (root/'README.md').write_text('# original\n## 标注约定\noriginal convention\n## 查看与验证\n')
        (root/'generator_snapshot').mkdir();Image.new('RGB',(10,10),'white').save(out/'boundary_preview.jpg')
        m.dump(out/'selection.json',dict(data_manifest_sha256=m.sha(root/'data_manifest.json'),records=records[:2],rule=m.RULE))
        return root,out,records

    def test_move_copy_links_split_and_repeat_filtering(self):
        with tempfile.TemporaryDirectory() as temp:
            root,out,records=self.fixture(temp);before={r[k]:m.sha(root/r[k]) for r in records for k in m.ASSETS}
            m.apply_partition(root,out)
            self.assertEqual(len(json.loads((root/'data_manifest.json').read_text())),2)
            self.assertEqual(len(json.loads((root/'train.json').read_text())),1)
            self.assertEqual(len(json.loads((root/'mini_hard/train.json').read_text())),2)
            self.assertTrue((root/records[0]['image']).is_symlink())
            self.assertFalse((root/records[0]['band_target']).is_symlink())
            for r in records[:2]:
                for key in m.ASSETS:self.assertEqual(m.sha(root/'mini_hard'/r[key]),before[r[key]])
            for r in records:
                for key in m.ASSETS:self.assertEqual(m.sha(root/r[key]),before[r[key]])
            m.apply_partition(root,out)  # completed applications are idempotent

    def test_failure_restores_all_original_bytes_and_asset_locations(self):
        with tempfile.TemporaryDirectory() as temp:
            root,out,records=self.fixture(temp);before={str(p.relative_to(root)):m.sha(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(m,'index_page',side_effect=RuntimeError('intentional rollback test')):
                with self.assertRaisesRegex(RuntimeError,'intentional'):m.apply_partition(root,out)
            for name,digest in before.items():self.assertEqual(m.sha(root/name),digest)
            self.assertFalse((root/'mini_hard').exists());self.assertFalse((root/'.mini_hard_partition.lock').exists())
            self.assertFalse((root/records[0]['image']).is_symlink())


if __name__=='__main__':unittest.main()
