import torch
import torch.nn as nn
from PIL import Image
from numpy.typing import NDArray
from collections.abc import Iterable

from .decoder import ImageTextFusion
from .prompt_aggregator import PromptAggregator

class BaseFusionHead(nn.Module):
    def __init__(self) -> None:
        pass
    
    def forward(
        self,
        images: Image.Image | NDArray | torch.Tensor | Iterable[Image.Image | NDArray | torch.Tensor],
    ):
        pass
