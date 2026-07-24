import torch.nn as nn

from flowsis.pretrained import DetectorArchitecture
from .base import FlowSISBase


class FlowSIS(nn.Module):
    def __init__(
        self,
        detector_name_or_path: str | None = None,
        *,
        detector_architecture: DetectorArchitecture | None = None,
        siglip2_name_or_path: str = "google/siglip2-base-patch16-224",
    ) -> None:
        super().__init__()
        self.base = FlowSISBase(
            detector_name_or_path,
            detector_architecture=detector_architecture,
            siglip2_name_or_path=siglip2_name_or_path,
        )
