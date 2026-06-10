from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_image_tokens(image_features: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    if image_features.ndim != 4:
        raise ValueError(
            "Expected image features with shape [B, C, H, W], "
            f"but received {tuple(image_features.shape)}."
        )

    _, _, height, width = image_features.shape
    tokens = image_features.flatten(2).transpose(1, 2)
    return tokens, (height, width)


def _to_image_grid(tokens: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
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


def _validate_feature_list(image_features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    feature_list = list(image_features)
    if not feature_list:
        raise ValueError("Expected at least one image feature map.")

    reference_batch, reference_channels = feature_list[0].shape[:2]
    for level, feature_map in enumerate(feature_list):
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

    return feature_list


def _resolve_activation(activation: str) -> nn.Module:
    if activation == "gelu":
        return nn.GELU()
    if activation == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation function: {activation}")


class ImageTextFusionBlock(nn.Module):
    """
    Fuse image-shaped features with text tokens while preserving the 2D layout.

    The module keeps `[B, C, H, W]` as the canonical image representation and
    flattens to `[B, H*W, C]` only inside the attention operations.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        act = _resolve_activation(activation)

        self.image_norm1 = nn.LayerNorm(d_model)
        self.image_norm2 = nn.LayerNorm(d_model)
        self.image_norm3 = nn.LayerNorm(d_model)
        self.text_norm = nn.LayerNorm(d_model)

        # Keep flatten/reshape local to the attention block so callers can work
        # with image-shaped tensors for dense prediction tasks like segmentation.
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_embeddings.ndim != 3:
            raise ValueError(
                "Expected text embeddings with shape [B, T, C], "
                f"but received {tuple(text_embeddings.shape)}."
            )

        image_tokens, spatial_shape = _to_image_tokens(image_features)
        normalized_image_tokens = self.image_norm1(image_tokens)
        self_attention_output, _ = self.self_attention(
            normalized_image_tokens,
            normalized_image_tokens,
            normalized_image_tokens,
            need_weights=False,
        )
        image_tokens = image_tokens + self.dropout1(self_attention_output)

        normalized_cross_query = self.image_norm2(image_tokens)
        normalized_text = self.text_norm(text_embeddings)
        cross_attention_output, _ = self.cross_attention(
            normalized_cross_query,
            normalized_text,
            normalized_text,
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )
        image_tokens = image_tokens + self.dropout2(cross_attention_output)

        ffn_output = self.ffn(self.image_norm3(image_tokens))
        image_tokens = image_tokens + self.dropout3(ffn_output)
        return _to_image_grid(image_tokens, spatial_shape)


class ImageTextFusion(nn.Module):
    """
    Stack multiple image/text fusion blocks and optionally predict mask logits.

    Accepts either a single `[B, C, H, W]` feature map or a multi-scale list of
    RT-DETRv2 encoder features ordered from highest to lowest resolution.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ImageTextFusionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.level_fuse = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            _resolve_activation(activation),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            _resolve_activation(activation),
            nn.Conv2d(d_model, out_channels, kernel_size=1),
        )

    def _fuse_single_scale(
        self,
        image_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused_features = image_features
        for block in self.blocks:
            fused_features = block(
                fused_features,
                text_embeddings,
                text_padding_mask=text_padding_mask,
            )
        return fused_features

    def _merge_multiscale_features(self, feature_list: Sequence[torch.Tensor]) -> torch.Tensor:
        validated_features = _validate_feature_list(feature_list)
        target_height, target_width = validated_features[0].shape[-2:]

        merged_features = validated_features[0]
        for feature_map in validated_features[1:]:
            merged_features = merged_features + F.interpolate(
                feature_map,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )

        merged_features = merged_features / len(validated_features)
        return self.level_fuse(merged_features)

    def forward(
        self,
        image_features: torch.Tensor | Sequence[torch.Tensor],
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
        *,
        return_mask_logits: bool = False,
    ) -> (
        torch.Tensor
        | list[torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[list[torch.Tensor], torch.Tensor]
    ):
        if isinstance(image_features, torch.Tensor):
            fused_features = self._fuse_single_scale(
                image_features,
                text_embeddings,
                text_padding_mask=text_padding_mask,
            )
            if not return_mask_logits:
                return fused_features

            mask_logits = self.mask_head(fused_features)
            return fused_features, mask_logits

        feature_list = _validate_feature_list(image_features)
        fused_feature_list = [
            self._fuse_single_scale(
                feature_map,
                text_embeddings,
                text_padding_mask=text_padding_mask,
            )
            for feature_map in feature_list
        ]
        if not return_mask_logits:
            return fused_feature_list

        merged_features = self._merge_multiscale_features(fused_feature_list)
        mask_logits = self.mask_head(merged_features)
        return fused_feature_list, mask_logits
