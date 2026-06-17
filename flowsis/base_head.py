from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn as nn

from .decoder import ImageTextFusion
from .mask_head import MaskHead
from .prompt_aggregator import ChannelAggregator


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
        activation: Literal["GELU", "RELU"],
        num_feature_levels,
        decode_pos_encode,
        image_self_attention,
        decode_window_size,
        use_shifted_windows,
        multiscale_merge,
        deformable_num_points,
        deformable_offset_scale,
        aggregator_dim,
        *,
        mask_head_hidden_dim: int | None = None,
        mask_output_dim: int = 1,
        mask_upsample_scales: tuple[int, ...] = (2, 2),
    ) -> None:
        super().__init__()
        self.mask_output_dim = int(mask_output_dim)

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
        self.channel_aggregator = ChannelAggregator(
            image_dim=image_dim,
            text_dim=text_dim,
            hidden_dim=aggregator_dim,
            output_dim=image_dim,
            dropout=dropout,
            activation=activation,
        )
        self.mask_head = MaskHead(
            image_dim=image_dim,
            text_dim=text_dim,
            hidden_dim=mask_head_hidden_dim or aggregator_dim,
            output_dim=mask_output_dim,
            upsample_scales=mask_upsample_scales,
            dropout=dropout,
            activation=activation,
        )

    def forward(
        self,
        multi_image_features: Iterable[torch.Tensor],
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
        *,
        mask_output_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        fused_feature_list, merged_features = self.decoder(
            multi_image_features,
            text_embeddings,
            text_padding_mask=text_padding_mask,
            return_merged_features=True,
        )

        channel_logits = self.channel_aggregator(
            merged_features,
            text_embeddings,
            text_padding_mask=text_padding_mask,
        )
        channel_gates = ChannelAggregator.compute_gates(channel_logits).unsqueeze(-1).unsqueeze(-1)
        gated_features = merged_features * channel_gates
        mask_logits = self.mask_head(
            gated_features,
            text_embeddings,
            text_padding_mask=text_padding_mask,
            output_size=mask_output_size,
        )

        if self.mask_output_dim == 1:
            mask_logits = mask_logits.squeeze(1)

        return {
            "fused_feature_list": fused_feature_list,
            "merged_features": merged_features,
            "gated_features": gated_features,
            "channel_logits": channel_logits,
            "channel_gates": channel_gates.squeeze(-1).squeeze(-1),
            "mask_logits": mask_logits,
        }
