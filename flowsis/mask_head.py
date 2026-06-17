from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowsis.decoder.shape import pool_text_embeddings
from flowsis.utils import resolve_activation


class _UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float,
        activation: Literal["GELU", "RELU"],
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            resolve_activation(activation),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            resolve_activation(activation),
        )

    def forward(self, features: torch.Tensor, *, scale_factor: int) -> torch.Tensor:
        upsampled = F.interpolate(
            features,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        return self.block(upsampled)


class MaskHead(nn.Module):
    """
    Text-conditioned decoder that maps fused image features to mask logits.

    The head first applies a global FiLM-style text modulation to the fused
    feature map, then progressively upsamples and refines the features before
    predicting the final mask logits.
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        *,
        hidden_dim: int | None = None,
        output_dim: int = 1,
        upsample_scales: Iterable[int] = (2, 2),
        dropout: float = 0.1,
        activation: Literal["GELU", "RELU"] = "GELU",
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, but received {output_dim}.")

        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim or image_dim)
        self.upsample_scales = tuple(int(scale) for scale in upsample_scales)

        if any(scale <= 0 for scale in self.upsample_scales):
            raise ValueError(
                "upsample_scales must only contain positive integers, "
                f"but received {self.upsample_scales}."
            )

        self.input_proj = nn.Sequential(
            nn.Conv2d(image_dim, self.hidden_dim, kernel_size=1),
            resolve_activation(activation),
        )
        self.text_affine = nn.Linear(text_dim, self.hidden_dim * 2)

        stage_dims = [self.hidden_dim]
        for _ in self.upsample_scales:
            next_dim = max(stage_dims[-1] // 2, self.output_dim * 4, 8)
            stage_dims.append(next_dim)

        self.blocks = nn.ModuleList(
            [
                _UpsampleBlock(
                    stage_dims[index],
                    stage_dims[index + 1],
                    dropout=dropout,
                    activation=activation,
                )
                for index in range(len(self.upsample_scales))
            ]
        )
        self.logit_head = nn.Conv2d(stage_dims[-1], self.output_dim, kernel_size=1)

    def _condition_features(
        self,
        features: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled_text = pool_text_embeddings(text_embeddings, text_padding_mask)
        if pooled_text.shape[0] != features.shape[0]:
            raise ValueError(
                "Text and image batches must match, "
                f"but got {pooled_text.shape[0]} and {features.shape[0]}."
            )

        gamma, beta = self.text_affine(pooled_text).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) # (B,_,1,1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gamma) + beta

    def forward(
        self,
        image_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if image_features.ndim != 4:
            raise ValueError(
                "Expected image features with shape [B, C, H, W], "
                f"but received {tuple(image_features.shape)}."
            )

        decoded_features = self.input_proj(image_features)
        decoded_features = self._condition_features(
            decoded_features,
            text_embeddings,
            text_padding_mask=text_padding_mask,
        )

        for scale_factor, block in zip(self.upsample_scales, self.blocks, strict=True):
            decoded_features = block(decoded_features, scale_factor=scale_factor)

        mask_logits = self.logit_head(decoded_features)
        if output_size is not None and mask_logits.shape[-2:] != output_size:
            mask_logits = F.interpolate(
                mask_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return mask_logits


__all__ = [
    "MaskHead",
]
