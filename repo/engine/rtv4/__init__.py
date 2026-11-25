"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

from .rtv4 import RTv4

from .matcher import HungarianMatcher
from .hybrid_encoder import HybridEncoder
from .dfine_decoder import DFINETransformer
from .rtdetrv2_decoder import RTDETRTransformerv2

from .postprocessor import PostProcessor
from .rtv4_criterion import RTv4Criterion
from .chem_model import ChemRTv4
from .chem_criterion import ChemCriterion, LineCriterion, LineHungarianMatcher
from .chem_postprocessor import ChemPostProcessor
from .chem_endpoint import (
    ChemRTv4Endpoint,
    EndpointLineTransformer,
    EndpointLineHungarianMatcher,
    EndpointLineCriterion,
    ChemEndpointCriterion,
    ChemEndpointPostProcessor,
)

from .dinov3_teacher import DINOv3TeacherModel
