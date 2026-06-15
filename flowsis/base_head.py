import torch
import torch.nn as nn
from PIL import Image
from typing import Literal
from numpy.typing import NDArray
from collections.abc import Iterable

from .decoder import ImageTextFusion
from .prompt_aggregator import PromptAggregator

class BaseFusionHead(nn.Module):
    def __init__(
        self,
        num_decode_layers: int,
        decode_embed_dim: int,
        image_dim: int,
        text_dim: int,
        nhead: int,
        decode_ffn_dim: int,
        dropout: float,
        activation: Literal["gelu", "relu"],
        num_feature_levels,
        decode_pos_encode,
        image_self_attention,
        decode_window_size,
        use_shifted_windows,
        multiscale_merge,
        deformable_num_points,
        deformable_offset_scale,
    ) -> None:
        self.decoder = ImageTextFusion(
            num_layers=num_decode_layers,
            embed_dim=decode_embed_dim,
            image_dim=image_dim,
            text_dim=text_dim,
            nhead=nhead,
            ffn_dim=decode_ffn_dim,
            dropout=dropout,
            activation=activation,
            num_feature_levels=num_feature_levels,
            pos_encode=decode_pos_encode,
            image_self_attention=image_self_attention,
            window_size=decode_window_size,
            use_shifted_windows=use_shifted_windows,
            multiscale_merge=multiscale_merge,
            deformable_num_points=deformable_num_points,
            deformable_offset_scale=deformable_offset_scale,
        )
    
    def forward(
        self,
        multi_image_features: Iterable[torch.Tensor],
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ):
        pass
