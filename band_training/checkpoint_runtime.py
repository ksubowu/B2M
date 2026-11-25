"""Build the base directly from a self-contained joint checkpoint.

Joint checkpoints already contain all base tensors; the historical config.base
path is provenance and need not point to an available parent checkpoint.
"""
from pathlib import Path
import sys


def load_joint_base(payload,device):
    root=Path(__file__).resolve().parents[1]
    if str(root/'repo') not in sys.path:sys.path.insert(0,str(root/'repo'))
    from engine.core import YAMLConfig
    config=YAMLConfig(str(root/'config/train_mp123_full.yml'))
    config.yaml_cfg['HGNetv2']['pretrained']=False
    base=config.model
    base.load_state_dict(payload['base'],strict=True)
    return base.to(device)
