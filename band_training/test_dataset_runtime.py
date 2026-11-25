import os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from .dataset_runtime import install_generator_path


class SeparateChemistryPackageTests(unittest.TestCase):
    def test_installed_package_needs_no_dataset_snapshot(self):
        with patch.dict(os.environ,{},clear=True),patch('importlib.util.find_spec',return_value=SimpleNamespace(origin='/external/chem_prep/__init__.py')):
            self.assertEqual(install_generator_path('/nonexistent/dataset'),Path('/external'))

    def test_missing_optional_package_explains_installation(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True),patch('importlib.util.find_spec',return_value=None):
            with self.assertRaisesRegex(ModuleNotFoundError,'chem-structure-prep'):
                install_generator_path(Path(tmp)/'dataset')

    def test_explicit_pinned_backend_overrides_installed_package(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(sys,'path',list(sys.path)):
            root=Path(tmp);(root/'chem_prep').mkdir();(root/'chem_prep/backend.py').write_text('')
            with patch.dict(os.environ,{'B2M_CHEM_PREP_PATH':tmp}):self.assertEqual(install_generator_path('/unused'),root)


if __name__=='__main__':unittest.main()
