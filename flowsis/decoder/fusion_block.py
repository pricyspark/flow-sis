import torch
import torch.nn as nn
from typing import Literal

from flowsis.utils import resolve_activation
from .shape import to_image_tokens, to_image_grid
from .window import (
    pad_to_window_size, 
    partition_padded_windows, 
    build_shifted_window_attention_mask, 
    build_padding_attention_mask, 
    merge_padded_windows,
)


class ImageTextFusionBlock(nn.Module):
    """
    Fuse image-shaped features with text tokens while preserving the 2D layout.

    The module keeps `[B, C, H, W]` as the canonical image representation and
    flattens to `[B, H*W, C]` only inside the attention operations.
    """

    def __init__(
        self,
        image_dim: int, # (B,HW,C)
        text_dim: int,
        nhead: int,
        ffn_dim: int = 2048,
        image_self_attention: Literal["GLOBAL", "WINDOW", "none"] = "GLOBAL",
        window_size: int = 8,
        window_shift_size: int = 0,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu"] = "gelu",
    ) -> None:
        super().__init__()
        
        # TODO: maybe add embed_dim param to reduce the number of
        # channels within each attention head.

        act = resolve_activation(activation)

        self.image_norm1 = nn.LayerNorm(image_dim)
        self.image_norm2 = nn.LayerNorm(image_dim)
        self.image_norm3 = nn.LayerNorm(image_dim)
        self.text_norm = nn.LayerNorm(text_dim)

        if image_self_attention not in {"GLOBAL", "WINDOW", "none"}:
            raise ValueError(
                "image_self_attention must be one of {'global', 'window', 'none'}, "
                f"but received {image_self_attention!r}."
            )
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, but received {window_size}.")
        if not 0 <= window_shift_size < window_size:
            raise ValueError(
                "window_shift_size must satisfy 0 <= window_shift_size < window_size, "
                f"but received {window_shift_size=} and {window_size=}."
            )

        self.image_self_attention = image_self_attention
        self.window_size = int(window_size)
        self.window_shift_size = int(window_shift_size)
        self.num_heads = int(nhead)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
            kdim=text_dim, # TODO:
            vdim=text_dim,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(image_dim, ffn_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, image_dim),
        )

    def _apply_window_self_attention(
        self,
        normalized_image_tokens: torch.Tensor,
        spatial_shape: tuple[int, int],
        positional_encoding: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized_image_grid = to_image_grid(normalized_image_tokens, spatial_shape)
        shift_size = self.window_shift_size
        if min(spatial_shape) <= self.window_size:
            shift_size = 0
        normalized_image_grid, original_shape, padded_shape = pad_to_window_size(
            normalized_image_grid,
            self.window_size,
        )
        if shift_size > 0:
            normalized_image_grid = torch.roll(
                normalized_image_grid,
                shifts=(-shift_size, -shift_size),
                dims=(-2, -1),
            )
        image_windows = partition_padded_windows(
            normalized_image_grid,
            self.window_size,
        )
        image_queries = image_windows
        image_keys = image_windows
        shift_mask = build_shifted_window_attention_mask(
            padded_shape,
            self.window_size,
            shift_size,
            device=normalized_image_grid.device,
        )

        padding_mask = build_padding_attention_mask(
            original_shape,
            padded_shape,
            self.window_size,
            shift_size,
            device=normalized_image_grid.device,
        )

        if shift_mask is None:
            attention_mask = padding_mask
        else:
            attention_mask = shift_mask | padding_mask

        if positional_encoding is not None:
            height, width = spatial_shape
            positional_grid = positional_encoding.reshape(1, height, width, -1).permute(0, 3, 1, 2)
            positional_grid = positional_grid.expand(normalized_image_grid.shape[0], -1, -1, -1)
            positional_grid, _, _ = pad_to_window_size(positional_grid, self.window_size)
            if shift_size > 0:
                positional_grid = torch.roll(
                    positional_grid,
                    shifts=(-shift_size, -shift_size),
                    dims=(-2, -1),
                )
            positional_windows = partition_padded_windows(positional_grid, self.window_size)
            image_queries = image_queries + positional_windows
            image_keys = image_keys + positional_windows

        batch_size = normalized_image_grid.shape[0]
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(batch_size, 1, 1)
            attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)

        window_outputs, _ = self.self_attention(
            image_queries,
            image_keys,
            image_windows,
            attn_mask=attention_mask,
            need_weights=False,
        )
        image_grid = merge_padded_windows(
            window_outputs,
            padded_shape,
            self.window_size,
        )

        if shift_size > 0:
            image_grid = torch.roll(
                image_grid,
                shifts=(shift_size, shift_size),
                dims=(-2, -1),
            )

        original_height, original_width = original_shape
        image_grid = image_grid[..., :original_height, :original_width]

        return image_grid.flatten(2).transpose(1, 2)

    def forward(
        self,
        image_features: torch.Tensor,   # (B,C,H,W)
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
        # image_tokens: (1,H*W,C)
        image_tokens, spatial_shape, positional_encoding = to_image_tokens(
            image_features,
            add_positional_encoding=add_positional_encoding,
        )
        normalized_image_tokens = self.image_norm1(image_tokens)
        if self.image_self_attention == "GLOBAL":
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
        elif self.image_self_attention == "WINDOW":
            self_attention_output = self._apply_window_self_attention(
                normalized_image_tokens,
                spatial_shape,
                positional_encoding,
            )
            image_tokens = image_tokens + self.dropout1(self_attention_output)

        image_cross_attn_query = self.image_norm2(image_tokens)
        if positional_encoding is not None:
            image_cross_attn_query = image_cross_attn_query + positional_encoding
        normalized_text = self.text_norm(text_embeddings)
        cross_attention_output, _ = self.cross_attention(
            image_cross_attn_query,
            normalized_text,
            normalized_text, # TODO: why is text the value and not image
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )
        image_tokens = image_tokens + self.dropout2(cross_attention_output)

        ffn_output = self.ffn(self.image_norm3(image_tokens))
        image_tokens = image_tokens + self.dropout3(ffn_output)
        return to_image_grid(image_tokens, spatial_shape)
