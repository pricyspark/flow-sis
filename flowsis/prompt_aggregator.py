import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from flowsis.utils import resolve_activation


class ImageConditionedPromptPooler(nn.Module):
    """Pool prompt vectors using inexpensive image-conditioned attention."""

    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

    @staticmethod
    def _pool_image(image_features: torch.Tensor) -> torch.Tensor:
        if image_features.ndim == 4:
            return image_features.mean(dim=(-2, -1))
        if image_features.ndim == 2:
            return image_features
        raise ValueError(
            "Expected image features with shape [B,C,H,W] or [B,C], "
            f"but received {tuple(image_features.shape)}."
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if text_embeddings.ndim == 2:
            return text_embeddings, None
        if text_embeddings.ndim != 3:
            raise ValueError(
                "Expected prompt embeddings with shape [B,P,D] or [B,D], "
                f"but received {tuple(text_embeddings.shape)}."
            )
        if (
            text_padding_mask is not None
            and text_padding_mask.shape != text_embeddings.shape[:2]
        ):
            raise ValueError("text_padding_mask must have shape [B,P].")

        image_state = F.normalize(
            self.image_proj(self._pool_image(image_features)), dim=-1
        )
        prompt_states = F.normalize(self.text_proj(text_embeddings), dim=-1)
        scores = torch.einsum("bph,bh->bp", prompt_states, image_state)

        if text_padding_mask is not None:
            valid = ~text_padding_mask
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1) * valid.to(scores.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            weights = torch.softmax(scores, dim=1)

        pooled = torch.einsum("bp,bpd->bd", weights, text_embeddings)
        return pooled, weights


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
        activation: Literal["gelu", "relu"] = "gelu",
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or max(image_dim, text_dim))
        output_dim = int(output_dim or image_dim)

        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.prompt_pooler = ImageConditionedPromptPooler(
            image_dim=image_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
        )
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

    def forward(
        self,
        image_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled_image = self._pool_image_features(image_features)
        pooled_text, _ = self.prompt_pooler(
            image_features,
            text_embeddings,
            text_padding_mask,
        )

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
    "ImageConditionedPromptPooler",
    "PromptAggregator",
]
