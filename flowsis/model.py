import torch
import torch.nn as nn
from PIL import Image
from numpy.typing import NDArray
from collections.abc import Iterable

from flowsis.utils import get_device
from flowsis.pretrained import DetectorArchitecture
from .base import FlowSISBase
from .temporal import TemporalRefinementBranch

class FlowSIS(nn.Module):
    def __init__(
        self,
        rt_detrv2_name_or_path: str = "PekingU/rtdetr_v2_r50vd",
        siglip2_name_or_path: str = "google/siglip2-base-patch16-224",
        detector_architecture: DetectorArchitecture = "rtdetrv2",
        detector_name_or_path: str | None = None,
    ) -> None:
        super().__init__()
        self.base = FlowSISBase(
            rt_detrv2_name_or_path,
            siglip2_name_or_path,
            detector_architecture,
            detector_name_or_path,
        )
    
    def forward(
        self,
        images: Image.Image | NDArray | torch.Tensor | Iterable[Image.Image | NDArray | torch.Tensor],
    ):
        pass
