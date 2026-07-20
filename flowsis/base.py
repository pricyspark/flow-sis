import torch
import torch.nn as nn
from PIL import Image
from numpy.typing import NDArray
from collections.abc import Iterable

from flowsis.pretrained import DetectorArchitecture, SigLIP2, load_detector
from flowsis.utils import get_device
from .base_head import BaseFusionHead

class FlowSISBase(nn.Module):
    def __init__(
        self,
        rt_detrv2_name_or_path: str = "PekingU/rtdetr_v2_r50vd",
        siglip2_name_or_path: str = "google/siglip2-base-patch16-224",
        detector_architecture: DetectorArchitecture = "rtdetrv2",
        detector_name_or_path: str | None = None,
    ) -> None:
        super().__init__()
        source = detector_name_or_path or rt_detrv2_name_or_path
        self.detector = load_detector(detector_architecture, source)
        self.siglip2 = SigLIP2.from_pretrained(siglip2_name_or_path)
        # self.head = BaseFusionHead()

    @property
    def rtdetrv2(self):
        """Backward-compatible name for the configured detector."""
        return self.detector
    
    def forward(
        self,
        images: Image.Image | NDArray | torch.Tensor | Iterable[Image.Image | NDArray | torch.Tensor],
    ):
        pass
