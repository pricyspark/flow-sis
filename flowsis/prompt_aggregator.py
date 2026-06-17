import torch
import torch.nn as nn
from typing import Literal

from flowsis.utils import resolve_activation


class ChannelAggregator(nn.Module):
    """
    Global text-conditioned channel scoring head.

    The module pools spatial image features, combines them with a pooled text
    embedding, and predicts sample-level logits that can be used as channel
    gates or expert-routing scores.
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        *,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        dropout: float = 0.1,
        activation: Literal["GELU", "RELU"] = "GELU",
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or max(image_dim, text_dim))
        output_dim = int(output_dim or image_dim)

        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            resolve_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def _pool_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        if image_features.ndim == 4:
            return image_features.flatten(2).mean(dim=-1)
        if image_features.ndim == 3:
            return image_features.mean(dim=-1)
        raise ValueError(
            "Expected image features with shape [B, C, H, W] or [B, C, N], "
            f"but received {tuple(image_features.shape)}."
        )

    def _pool_text_embeddings(
        self,
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

    def forward(
        self,
        image_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled_image = self._pool_image_features(image_features)
        pooled_text = self._pool_text_embeddings(text_embeddings, text_padding_mask)

        if pooled_image.shape[0] != pooled_text.shape[0]:
            raise ValueError(
                "Image and text batches must match, "
                f"but got {pooled_image.shape[0]} and {pooled_text.shape[0]}."
            )

        image_state = self.image_proj(pooled_image)
        text_state = self.text_proj(pooled_text)
        logits = self.scorer(torch.cat([image_state, text_state], dim=-1))
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits

    @staticmethod
    def compute_weights(logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return torch.softmax(logits, dim=dim)

    @staticmethod
    def compute_gates(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)


PromptAggregator = ChannelAggregator

__all__ = [
    "ChannelAggregator",
    "PromptAggregator",
]
