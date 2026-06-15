import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from collections.abc import Iterable

from flowsis.utils import resolve_activation
from .fusion_block import ImageTextFusionBlock
from .deformable import TextGuidedDeformableFusion
from .shape import validate_feature_list

class ImageTextFusion(nn.Module):
    """
    Stack multiple image/text fusion blocks and optionally predict mask logits.

    Accepts either a single `[B, C, H, W]` feature map or a multi-scale list of
    RT-DETRv2 encoder features ordered from highest to lowest resolution.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        image_dim: int,
        text_dim: int,
        nhead: int,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu"] = "gelu",
        num_feature_levels: int = 3,
        pos_encode: Literal["NONE", "FIRST", "SECOND", "ALL"] = "FIRST",
        image_self_attention: Literal["global", "window", "none"] = "global",
        window_size: int = 8,
        use_shifted_windows: bool = True,
        multiscale_merge: Literal["conv", "deformable"] = "conv",
        deformable_num_points: int = 4,
        deformable_offset_scale: float = 2.0,
    ) -> None:
        super().__init__()

        self.num_feature_levels = int(num_feature_levels)
        if multiscale_merge not in {"conv", "deformable"}:
            raise ValueError(
                "multiscale_merge must be one of {'conv', 'deformable'}, "
                f"but received {multiscale_merge!r}."
            )

        self.blocks = nn.ModuleList(
            [
                ImageTextFusionBlock(
                    image_dim=image_dim,
                    text_dim=text_dim,
                    nhead=nhead,
                    ffn_dim=ffn_dim,
                    image_self_attention=image_self_attention,
                    window_size=window_size,
                    window_shift_size=(
                        window_size // 2
                        if use_shifted_windows and image_self_attention == "window" and layer_index % 2 == 1
                        else 0
                    ),
                    dropout=dropout,
                    activation=activation,
                )
                for layer_index in range(num_layers)
            ]
        )
        self.multiscale_merge = multiscale_merge
        self.level_embedding = nn.Embedding(self.num_feature_levels, embed_dim)
        self.level_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * self.num_feature_levels, embed_dim, kernel_size=1),
            resolve_activation(activation),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            resolve_activation(activation),
        )
        self.deformable_fuse = None
        if self.multiscale_merge == "deformable":
            self.deformable_fuse = TextGuidedDeformableFusion(
                image_dim=image_dim,
                text_dim=text_dim,
                num_feature_levels=self.num_feature_levels,
                num_points=deformable_num_points,
                offset_scale=deformable_offset_scale,
                dropout=dropout,
                activation=activation,
            )

        pos_encode_dict = {"NONE": -1, "FIRST": 0, "SECOND": 1, "ALL": float("inf")}
        self.pos_encode_blocks = pos_encode_dict[pos_encode]

    def _fuse_single_scale(
        self,
        image_features: torch.Tensor,   # (B,C,H,W)
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
        *,
        level_index: int,
    ) -> torch.Tensor:
        if not 0 <= level_index < self.num_feature_levels:
            raise ValueError(
                f"level_index must be in [0, {self.num_feature_levels}), "
                f"but received {level_index}."
            )

        level_bias = self.level_embedding.weight[level_index].view(1, -1, 1, 1)
        fused_features = image_features + level_bias
        for block_index, block in enumerate(self.blocks):
            add_positional_encoding = (
                block_index == self.pos_encode_blocks 
                or self.pos_encode_blocks == float("inf")
            )
            fused_features = block(
                fused_features,
                text_embeddings,
                text_padding_mask=text_padding_mask,
                add_positional_encoding=add_positional_encoding,
            )
        return fused_features

    def _merge_multiscale_features(self, feature_list: Iterable[torch.Tensor]) -> torch.Tensor:
        validated_features = validate_feature_list(feature_list)
        if len(validated_features) != self.num_feature_levels:
            raise ValueError(
                f"Expected {self.num_feature_levels} feature levels, "
                f"but received {len(validated_features)}."
            )
        target_height, target_width = validated_features[0].shape[-2:]

        resized_features = [validated_features[0]]
        for feature_map in validated_features[1:]:
            resized_features.append(
                F.interpolate(
                    feature_map,
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                )
            )

        merged_features = torch.cat(resized_features, dim=1) # (B,C*3,H,W)
        return self.level_fuse(merged_features) # (B,C,H,W)

    def forward(
        self,
        multi_image_features: Iterable[torch.Tensor],
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
        *,
        return_mask_logits: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], torch.Tensor]:
        feature_list = validate_feature_list(multi_image_features)
        fused_feature_list = [
            self._fuse_single_scale(
                feature_map,
                text_embeddings,
                text_padding_mask=text_padding_mask,
                level_index=level_index,
            )
            for level_index, feature_map in enumerate(feature_list)
        ]
        if not return_mask_logits:
            return fused_feature_list

        if self.deformable_fuse is not None:
            merged_features = self.deformable_fuse(
                fused_feature_list,
                text_embeddings,
                text_padding_mask=text_padding_mask,
            )
        else:
            merged_features = self._merge_multiscale_features(fused_feature_list)
        return fused_feature_list, merged_features
