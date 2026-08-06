from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .position import build_2d_sincos_pos_encoding


def to_image_tokens(
    image_features: torch.Tensor,  # (B,C,H,W)
    *,
    add_positional_encoding: bool = False,
) -> tuple[torch.Tensor, tuple[int, int], torch.Tensor | None]:
    if image_features.ndim != 4:
        raise ValueError(
            "Expected image features with shape [B, C, H, W], "
            f"but received {tuple(image_features.shape)}."
        )

    _, channels, height, width = image_features.shape
    tokens = image_features.flatten(2).transpose(1, 2)  # (B,H*W,C)
    positional_encoding = None
    if add_positional_encoding:
        positional_encoding = build_2d_sincos_pos_encoding(
            height,
            width,
            channels,
            device=image_features.device,
            dtype=image_features.dtype,
        )  # (1,H*W,C)
    return tokens, (height, width), positional_encoding


def to_image_grid(tokens: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(
            "Expected token tensor with shape [B, N, C], "
            f"but received {tuple(tokens.shape)}."
        )

    height, width = spatial_shape
    batch_size, num_tokens, channels = tokens.shape
    expected_tokens = height * width
    if num_tokens != expected_tokens:
        raise ValueError(
            f"Token count {num_tokens} does not match spatial shape "
            f"{spatial_shape} ({expected_tokens} locations)."
        )

    return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


def pool_text_embeddings(
    text_embeddings: torch.Tensor,
    text_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if text_embeddings.ndim == 2:
        return text_embeddings
    if text_embeddings.ndim != 3:
        raise ValueError(
            "Expected text embeddings with shape [B, D] or [B, T, D], "
            f"but received {tuple(text_embeddings.shape)}."
        )

    if text_padding_mask is None:
        return text_embeddings.mean(dim=1)

    if text_padding_mask.shape != text_embeddings.shape[:2]:
        raise ValueError(
            "text_padding_mask must match the first two text dimensions, "
            f"but got mask {tuple(text_padding_mask.shape)} and embeddings "
            f"{tuple(text_embeddings.shape)}."
        )

    valid_mask = (~text_padding_mask).unsqueeze(-1).to(text_embeddings.dtype)
    valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
    return (text_embeddings * valid_mask).sum(dim=1) / valid_count


def validate_feature_list(
    multi_image_features: Iterable[torch.Tensor],
) -> list[torch.Tensor]:
    multi_image_features = list(multi_image_features)
    if not multi_image_features:
        raise ValueError("Expected at least one image feature map.")

    reference_batch, reference_channels = multi_image_features[0].shape[:2]
    for level, feature_map in enumerate(multi_image_features):
        if feature_map.ndim != 4:
            raise ValueError(
                "Expected every feature map to have shape [B, C, H, W], "
                f"but level {level} has shape {tuple(feature_map.shape)}."
            )

        batch_size, channels = feature_map.shape[:2]
        if batch_size != reference_batch or channels != reference_channels:
            raise ValueError(
                "Expected all feature maps to share the same batch size and channel count, "
                f"but level 0 has [B={reference_batch}, C={reference_channels}] and "
                f"level {level} has [B={batch_size}, C={channels}]."
            )

    return multi_image_features
