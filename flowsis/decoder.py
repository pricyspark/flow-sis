from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_2d_sincos_pos_encoding(
    height: int,
    width: int,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if channels % 4 != 0:
        raise ValueError(
            "2D sinusoidal positional encoding requires channels divisible by 4, "
            f"but received channels={channels}."
        )
        
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height=} and {width=}.")
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels=}.")

    quarter_channels = channels // 4
    frequencies = torch.arange(quarter_channels, device=device, dtype=torch.float32)
    frequencies = 1.0 / (10000 ** (frequencies / quarter_channels))
    
    y_coords = torch.arange(height, device=device, dtype=torch.float32)
    x_coords = torch.arange(width, device=device, dtype=torch.float32)

    y_angles = y_coords[:, None] * frequencies[None, :]
    x_angles = x_coords[:, None] * frequencies[None, :]

    y_encoding = torch.cat([y_angles.sin(), y_angles.cos()], dim=1)
    x_encoding = torch.cat([x_angles.sin(), x_angles.cos()], dim=1)

    y_encoding = y_encoding[:, None, :].expand(height, width, channels // 2)
    x_encoding = x_encoding[None, :, :].expand(height, width, channels // 2)
    
    positional_encoding = torch.cat([y_encoding, x_encoding], dim=-1)
    positional_encoding = positional_encoding.reshape(1, height * width, channels)
    
    return positional_encoding.to(dtype=dtype)


def _to_image_tokens(
    image_features: torch.Tensor,
    *,
    add_positional_encoding: bool = False,
) -> tuple[torch.Tensor, tuple[int, int], torch.Tensor | None]:
    if image_features.ndim != 4:
        raise ValueError(
            "Expected image features with shape [B, C, H, W], "
            f"but received {tuple(image_features.shape)}."
        )

    _, channels, height, width = image_features.shape
    tokens = image_features.flatten(2).transpose(1, 2)
    positional_encoding = None
    if add_positional_encoding:
        positional_encoding = _build_2d_sincos_pos_encoding(
            height,
            width,
            channels,
            device=image_features.device,
            dtype=image_features.dtype,
        )
    return tokens, (height, width), positional_encoding


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


def _validate_feature_list(image_features: Iterable[torch.Tensor]) -> list[torch.Tensor]:
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
        *,
        add_positional_encoding: bool = False,
    ) -> torch.Tensor:
        if text_embeddings.ndim != 3:
            raise ValueError(
                "Expected text embeddings with shape [B, T, C], "
                f"but received {tuple(text_embeddings.shape)}."
            )

        image_tokens, spatial_shape, positional_encoding = _to_image_tokens(
            image_features,
            add_positional_encoding=add_positional_encoding,
        )
        normalized_image_tokens = self.image_norm1(image_tokens)
        image_self_attn_query = normalized_image_tokens
        image_self_attn_key = normalized_image_tokens
        if positional_encoding is not None:
            image_self_attn_query = image_self_attn_query + positional_encoding
            image_self_attn_key = image_self_attn_key + positional_encoding

        self_attention_output, _ = self.self_attention(
            image_self_attn_query,
            image_self_attn_key,
            normalized_image_tokens,
            need_weights=False,
        )
        image_tokens = image_tokens + self.dropout1(self_attention_output)

        image_cross_attn_query = self.image_norm2(image_tokens)
        if positional_encoding is not None:
            image_cross_attn_query = image_cross_attn_query + positional_encoding
        normalized_text = self.text_norm(text_embeddings)
        cross_attention_output, _ = self.cross_attention(
            image_cross_attn_query,
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
        num_feature_levels: int = 3,
    ) -> None:
        super().__init__()

        self.num_feature_levels = int(num_feature_levels)
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
        self.level_embedding = nn.Embedding(self.num_feature_levels, d_model)
        self.level_fuse = nn.Sequential(
            nn.Conv2d(d_model * self.num_feature_levels, d_model, kernel_size=1),
            _resolve_activation(activation),
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
            fused_features = block(
                fused_features,
                text_embeddings,
                text_padding_mask=text_padding_mask,
                add_positional_encoding=block_index == 0,
            )
        return fused_features

    def _merge_multiscale_features(self, feature_list: Sequence[torch.Tensor]) -> torch.Tensor:
        validated_features = _validate_feature_list(feature_list)
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

        merged_features = torch.cat(resized_features, dim=1)
        return self.level_fuse(merged_features)

    def forward(
        self,
        image_features: Iterable[torch.Tensor],
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
        feature_list = _validate_feature_list(image_features)
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

        merged_features = self._merge_multiscale_features(fused_feature_list)
        mask_logits = self.mask_head(merged_features)
        return fused_feature_list, mask_logits
