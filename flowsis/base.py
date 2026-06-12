import torch
import torch.nn as nn
from PIL import Image
from numpy.typing import NDArray
from collections.abc import Iterable

from flowsis.pretrained import RTDetrV2, SigLIP2
from flowsis.utils import get_device
from .base_head import BaseFusionHead

class FlowSISBase(nn.Module):
    def __init__(
        self,
        rt_detrv2_name_or_path: str = "PekingU/rtdetr_v2_r18vd",
        siglip2_name_or_path: str = "google/siglip2-base-patch16-224",
    ) -> None:
        self.rtdetrv2 = RTDetrV2.from_pretrained(rt_detrv2_name_or_path)
        self.siglip2 = SigLIP2.from_pretrained(siglip2_name_or_path)
        # self.head = BaseFusionHead()
    
    def forward(
        self,
        images: Image.Image | NDArray | torch.Tensor | Iterable[Image.Image | NDArray | torch.Tensor],
    ):
        pass
