import torch
import torch.nn as nn
from PIL import Image
from numpy.typing import NDArray
from collections.abc import Iterable


class TemporalRefinementBranch(nn.Module):
    def __init__(self) -> None:
        pass
    
    def forward(
        self,
        images: Image.Image | NDArray | torch.Tensor | Iterable[Image.Image | NDArray | torch.Tensor],
        prev_images: Image.Image | NDArray | torch.Tensor | Iterable[Image.Image | NDArray | torch.Tensor],
    ):
        pass
