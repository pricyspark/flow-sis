import torch
import torch.nn as nn
from PIL import Image
from numpy.typing import NDArray
from collections.abc import Iterable

from flowsis.utils import get_device
from .base import FlowSISBase
from .temporal import TemporalRefinementBranch

class FlowSIS(nn.Module):
    def __init__(
        self,
        rt_detrv2_name_or_path: str = "PekingU/rtdetr_v2_r18vd",
        siglip2_name_or_path: str = "google/siglip2-base-patch16-224",
    ) -> None:
        self.base = FlowSISBase(
            rt_detrv2_name_or_path,
            siglip2_name_or_path,
        )
    
    def forward(
        self,
        images: Image.Image | NDArray | torch.Tensor | Iterable[Image.Image | NDArray | torch.Tensor],
    ):
        pass
