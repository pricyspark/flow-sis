from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn as nn

from .decoder import ImageTextFusion
from .mask_head import MaskHead
from .prompt_aggregator import ChannelAggregator


class BaseFusionHead(nn.Module):
    """
    Compose the fusion decoder, optional channel modulation, and mask head.

    The decoder is responsible for per-level image/text fusion and may also
    optionally collapse the multi-scale feature list into a single feature map.
    `BaseFusionHead` then selects the mask input feature map, optionally applies
    a text-conditioned channel modulation step, and predicts the final mask.
    """

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
        num_feature_levels: int,
        decode_pos_encode: Literal["none", "first", "second", "all"],
        image_self_attention: Literal["GLOBAL", "WINDOW", "none"],
        decode_window_size: int,
        use_shifted_windows: bool,
        multiscale_merge: Literal["conv", "deformable", "none"],
        deformable_num_points: int,
        deformable_offset_scale: float,
        aggregator_dim: int | None = None,
        *,
        channel_aggregation: Literal["none", "sigmoid", "softmax"] = "sigmoid",
        mask_feature_source: Literal["merged", "highest_resolution"] = "merged",
        mask_head_hidden_dim: int | None = None,
        mask_output_dim: int = 1,
        mask_upsample_scales: tuple[int, ...] = (2, 2),
    ) -> None:
        super().__init__()

        if channel_aggregation not in {"none", "sigmoid", "softmax"}:
            raise ValueError(
                "channel_aggregation must be one of {'none', 'sigmoid', 'softmax'}, "
                f"but received {channel_aggregation!r}."
            )
        if mask_feature_source not in {"merged", "highest_resolution"}:
            raise ValueError(
                "mask_feature_source must be one of {'merged', 'highest_resolution'}, "
                f"but received {mask_feature_source!r}."
            )
        if mask_feature_source == "merged" and multiscale_merge == "none":
            raise ValueError(
                "mask_feature_source='merged' requires multiscale_merge to produce a "
                "merged feature map. Use multiscale_merge='conv' or 'deformable', or "
                "switch mask_feature_source to 'highest_resolution'."
            )

        self.channel_aggregation = channel_aggregation
        self.mask_feature_source = mask_feature_source
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
        self.channel_aggregator = None
        if self.channel_aggregation != "none":
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
            hidden_dim=mask_head_hidden_dim if mask_head_hidden_dim is not None else aggregator_dim,
            output_dim=mask_output_dim,
            upsample_scales=mask_upsample_scales,
            dropout=dropout,
            activation=activation,
        )

    def _select_mask_features(
        self,
        fused_feature_list: list[torch.Tensor],
        merged_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.mask_feature_source == "highest_resolution":
            return fused_feature_list[0]
        if merged_features is None:
            raise RuntimeError(
                "Expected merged decoder features, but the decoder returned None."
            )
        return merged_features

    def _apply_channel_aggregation(
        self,
        image_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.channel_aggregator is None:
            return image_features, None, None

        channel_logits = self.channel_aggregator(
            image_features,
            text_embeddings,
            text_padding_mask=text_padding_mask,
        )
        if self.channel_aggregation == "sigmoid":
            channel_modulation = ChannelAggregator.compute_gates(channel_logits)
        else:
            channel_modulation = ChannelAggregator.compute_weights(channel_logits, dim=-1)

        modulated_features = image_features * channel_modulation.unsqueeze(-1).unsqueeze(-1)
        return modulated_features, channel_logits, channel_modulation

    def forward(
        self,
        multi_image_features: Iterable[torch.Tensor],
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
        *,
        mask_output_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        fused_feature_list, merged_features = self.decoder(
            multi_image_features,
            text_embeddings,
            text_padding_mask=text_padding_mask,
            return_merged_features=True,
        )

        mask_features = self._select_mask_features(fused_feature_list, merged_features)
        modulated_features, channel_logits, channel_modulation = self._apply_channel_aggregation(
            mask_features,
            text_embeddings,
            text_padding_mask=text_padding_mask,
        )
        mask_logits = self.mask_head(
            modulated_features,
            text_embeddings,
            text_padding_mask=text_padding_mask,
            output_size=mask_output_size,
        )

        if self.mask_output_dim == 1:
            mask_logits = mask_logits.squeeze(1)

        return {
            "fused_feature_list": fused_feature_list,
            "merged_features": merged_features,
            "mask_features": mask_features,
            "modulated_features": modulated_features,
            "channel_logits": channel_logits,
            "channel_modulation": channel_modulation,
            "mask_logits": mask_logits,
        }
