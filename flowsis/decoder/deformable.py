from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowsis.utils import resolve_activation
from .shape import validate_feature_list, pool_text_embeddings
from .position import build_reference_grid

class TextGuidedDeformableFusion(nn.Module):
    """
    Lightweight multi-scale deformable fusion built from standard PyTorch ops.

    This is intentionally a preliminary implementation: each high-resolution
    query location predicts a small set of sampling offsets over every feature
    level, and the module aggregates the sampled values with learned weights.
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        num_feature_levels: int,
        *,
        num_points: int = 4,
        offset_scale: float = 2.0,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu"] = "gelu",
    ) -> None:
        super().__init__()

        if num_feature_levels <= 0:
            raise ValueError(
                f"num_feature_levels must be positive, but received {num_feature_levels}."
            )
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, but received {num_points}.")
        if offset_scale <= 0:
            raise ValueError(f"offset_scale must be positive, but received {offset_scale}.")

        self.num_feature_levels = int(num_feature_levels)
        self.num_points = int(num_points)
        self.offset_scale = float(offset_scale)

        self.query_proj = nn.Conv2d(image_dim, image_dim, kernel_size=1)
        self.text_proj = nn.Linear(text_dim, image_dim)
        self.value_proj = nn.ModuleList(
            [nn.Conv2d(image_dim, image_dim, kernel_size=1) for _ in range(self.num_feature_levels)]
        )
        self.offset_head = nn.Conv2d(
            image_dim,
            self.num_feature_levels * self.num_points * 2,
            kernel_size=3,
            padding=1,
        )
        self.attention_head = nn.Conv2d(
            image_dim,
            self.num_feature_levels * self.num_points,
            kernel_size=3,
            padding=1,
        )
        self.output_proj = nn.Sequential(
            nn.Conv2d(image_dim, image_dim, kernel_size=1),
            resolve_activation(activation),
            nn.Dropout2d(dropout),
            nn.Conv2d(image_dim, image_dim, kernel_size=1),
        )

    def _sample_level(
        self,
        value_features: torch.Tensor,
        sampling_locations: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels = value_features.shape[:2]
        query_height, query_width, num_points = sampling_locations.shape[1:4]
        sampling_grid = sampling_locations.mul(2.0).sub(1.0).reshape(
            batch_size,
            query_height,
            query_width * num_points,
            2,
        )
        sampled = F.grid_sample(
            value_features,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.reshape(batch_size, channels, query_height, query_width, num_points)

    def forward(
        self,
        feature_list: Iterable[torch.Tensor],
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validated_features = validate_feature_list(feature_list)
        if len(validated_features) != self.num_feature_levels:
            raise ValueError(
                f"Expected {self.num_feature_levels} feature levels, "
                f"but received {len(validated_features)}."
            )

        query_features = validated_features[0]
        batch_size, _, query_height, query_width = query_features.shape
        pooled_text = pool_text_embeddings(text_embeddings, text_padding_mask)
        if pooled_text.shape[0] != batch_size:
            raise ValueError(
                "Text and image batches must match, "
                f"but got {pooled_text.shape[0]} and {batch_size}."
            )

        text_bias = self.text_proj(pooled_text).view(batch_size, -1, 1, 1)
        query_state = self.query_proj(query_features) + text_bias

        sampling_offsets = self.offset_head(query_state)
        sampling_offsets = sampling_offsets.permute(0, 2, 3, 1).reshape(
            batch_size,
            query_height,
            query_width,
            self.num_feature_levels,
            self.num_points,
            2,
        )
        attention_logits = self.attention_head(query_state).permute(0, 2, 3, 1).reshape(
            batch_size,
            query_height,
            query_width,
            self.num_feature_levels * self.num_points,
        )
        attention_weights = torch.softmax(attention_logits, dim=-1).reshape(
            batch_size,
            query_height,
            query_width,
            self.num_feature_levels,
            self.num_points,
        )

        reference_grid = build_reference_grid(
            query_height,
            query_width,
            device=query_features.device,
            dtype=query_features.dtype,
        ).expand(batch_size, -1, -1, -1)
        aggregated = torch.zeros_like(query_features)

        for level_index, feature_map in enumerate(validated_features):
            value_features = self.value_proj[level_index](feature_map)
            offset_normalizer = query_features.new_tensor(
                [feature_map.shape[-1], feature_map.shape[-2]]
            ).view(1, 1, 1, 1, 2)
            level_offsets = torch.tanh(sampling_offsets[:, :, :, level_index])
            level_offsets = level_offsets * (self.offset_scale / offset_normalizer)
            sampling_locations = (reference_grid.unsqueeze(3) + level_offsets).clamp(0.0, 1.0)
            sampled = self._sample_level(value_features, sampling_locations)
            level_weights = attention_weights[:, :, :, level_index].unsqueeze(1)
            aggregated = aggregated + (sampled * level_weights).sum(dim=-1)

        return query_features + self.output_proj(aggregated)
