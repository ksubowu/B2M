# Third-party notices

The root MIT license applies to original B2M contributions. It does not
replace the licenses or copyright notices of bundled third-party components.

- **Detection engine and inherited model code** (`repo/engine`, `repo/configs`,
  and inherited portions of `repo/`): the supplied upstream source carries
  DEIM, D-FINE, RT-DETR, and other attribution headers. Its supplied
  Apache License 2.0 is retained verbatim at `repo/LICENSE`. Existing notices
  are preserved. B2M contains chemistry-specific adaptations of that engine;
  those derivative files remain subject to the supplied Apache-2.0 terms.
  The publication snapshot changes B2M application entry points and launchers;
  it does not modify the copied detection engine files.
- **Ketcher browser assets** (`repo/i2bl_webapp/ketcher`): bundled editor assets
  and dependency license texts are preserved, including `*.LICENSE.txt` files.
  These assets are not relicensed by B2M's root MIT license.

The Indigo/MolScribe renderer, fonts, and dataset-generation code are maintained
in the separate [chem-structure-prep](https://github.com/ksubowu/chem-structure-prep)
repository. They are not bundled in this B2M source tree. Optional complete
molecular graph evaluation imports that separately installed chemistry package;
its own third-party notices apply to those components.

Model checkpoints, OCR model files, molecular datasets, human annotations, and
experimental example images are not included in this source publication.
Their redistribution rights are separate from the code license.
