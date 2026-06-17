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
    Stack multiple image/text fusion blocks and optionally merge scales.

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
        pos_encode: Literal["none", "first", "second", "all"] = "first",
        image_self_attention: Literal["GLOBAL", "WINDOW", "none"] = "GLOBAL",
        window_size: int = 8,
        use_shifted_windows: bool = True,
        multiscale_merge: Literal["conv", "deformable", "none"] = "conv",
        deformable_num_points: int = 4,
        deformable_offset_scale: float = 2.0,
    ) -> None:
        super().__init__()

        self.num_feature_levels = int(num_feature_levels)
        self.feature_dim = int(image_dim)
        if int(embed_dim) != self.feature_dim:
            raise ValueError(
                "embed_dim must match image_dim in the current ImageTextFusion "
                f"implementation, but received embed_dim={embed_dim} and "
                f"image_dim={image_dim}."
            )
        if multiscale_merge not in {"conv", "deformable", "none"}:
            raise ValueError(
                "multiscale_merge must be one of {'conv', 'deformable', 'none'}, "
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
                        if use_shifted_windows and image_self_attention == "WINDOW" and layer_index % 2 == 1
                        else 0
                    ),
                    dropout=dropout,
                    activation=activation,
                )
                for layer_index in range(num_layers)
            ]
        )
        self.multiscale_merge = multiscale_merge
        self.level_embedding = nn.Embedding(self.num_feature_levels, self.feature_dim)
        self.level_fuse = None
        if self.multiscale_merge == "conv":
            self.level_fuse = nn.Sequential(
                nn.Conv2d(
                    self.feature_dim * self.num_feature_levels,
                    self.feature_dim,
                    kernel_size=1,
                ),
                resolve_activation(activation),
                nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
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

        pos_encode_dict = {"none": -1, "first": 0, "second": 1, "all": float("inf")}
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
        if self.level_fuse is None:
            raise RuntimeError("Conv multiscale fusion is not enabled for this decoder.")

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

    def _get_merged_features(
        self,
        fused_feature_list: Iterable[torch.Tensor],
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.multiscale_merge == "none":
            return None

        if self.deformable_fuse is not None:
            return self.deformable_fuse(
                fused_feature_list,
                text_embeddings,
                text_padding_mask=text_padding_mask,
            )

        return self._merge_multiscale_features(fused_feature_list)

    def forward(
        self,
        multi_image_features: Iterable[torch.Tensor],
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
        *,
        return_merged_features: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], torch.Tensor | None]:
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
        if not return_merged_features:
            return fused_feature_list

        merged_features = self._get_merged_features(
            fused_feature_list,
            text_embeddings,
            text_padding_mask=text_padding_mask,
        )
        return fused_feature_list, merged_features
