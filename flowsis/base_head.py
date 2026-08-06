from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import ImageTextFusion
from .prompt_aggregator import ChannelAggregator, ImageConditionedPromptPooler
from .utils import resolve_activation


def _make_conv(
    in_channels: int,
    out_channels: int,
    *,
    convolution: Literal["standard", "depthwise_separable"],
) -> nn.Module:
    if convolution == "standard":
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    if convolution == "depthwise_separable":
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
    raise ValueError(f"Unsupported convolution type {convolution!r}.")


class _UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float,
        activation: Literal["gelu", "relu"],
        convolution: Literal["standard", "depthwise_separable"],
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _make_conv(in_channels, out_channels, convolution=convolution),
            resolve_activation(activation),
            nn.Dropout2d(dropout),
            _make_conv(out_channels, out_channels, convolution=convolution),
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
        activation: Literal["gelu", "relu"] = "gelu",
        convolution: Literal["standard", "depthwise_separable"] = "depthwise_separable",
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
        nn.init.zeros_(self.text_affine.weight)
        nn.init.zeros_(self.text_affine.bias)
        self.prompt_pooler = ImageConditionedPromptPooler(
            image_dim=self.hidden_dim,
            text_dim=text_dim,
            hidden_dim=self.hidden_dim,
        )

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
                    convolution=convolution,
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
        pooled_text, _ = self.prompt_pooler(
            features,
            text_embeddings,
            text_padding_mask,
        )
        if pooled_text.shape[0] != features.shape[0]:
            raise ValueError(
                "Text and image batches must match, "
                f"but got {pooled_text.shape[0]} and {features.shape[0]}."
            )

        gamma, beta = self.text_affine(pooled_text).chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B,_,1,1)
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


class BaseFusionHead(nn.Module):
    """
    Compose the fusion decoder, optional channel modulation, and mask head.

    The decoder is responsible for per-level image/text fusion and may also
    optionally collapse the multi-scale feature list into a single feature map.
    `BaseFusionHead` then selects the mask input feature map, optionally applies
    a text-conditioned channel modulation step, and predicts the final mask.
    """

    def __init__(
        self,
        num_decode_layers: int,
        decode_embed_dim: int,
        image_dim: int,
        text_dim: int,
        nhead: int,
        decode_ffn_dim: int,
        dropout: float,
        activation: Literal["gelu", "relu"],
        num_feature_levels: int,
        decode_pos_encode: Literal["none", "first", "second", "all"],
        image_self_attention: Literal["GLOBAL", "WINDOW", "none"],
        decode_window_size: int,
        use_shifted_windows: bool,
        multiscale_merge: Literal["conv", "deformable", "none"],
        deformable_num_points: int,
        deformable_offset_scale: float,
        aggregator_dim: int | None = None,
        *,
        conv_merge_refinement: Literal["standard", "depthwise"] = "depthwise",
        channel_aggregation: Literal["none", "sigmoid", "softmax"] = "sigmoid",
        mask_feature_source: Literal["merged", "highest_resolution"] = "merged",
        mask_head_hidden_dim: int | None = None,
        mask_output_dim: int = 1,
        mask_upsample_scales: tuple[int, ...] = (2, 2),
        mask_convolution: Literal[
            "standard", "depthwise_separable"
        ] = "depthwise_separable",
    ) -> None:
        super().__init__()

        if channel_aggregation not in {"none", "sigmoid", "softmax"}:
            raise ValueError(
                "channel_aggregation must be one of {'none', 'sigmoid', 'softmax'}, "
                f"but received {channel_aggregation!r}."
            )
        if mask_feature_source not in {"merged", "highest_resolution"}:
            raise ValueError(
                "mask_feature_source must be one of {'merged', 'highest_resolution'}, "
                f"but received {mask_feature_source!r}."
            )
        if mask_feature_source == "merged" and multiscale_merge == "none":
            raise ValueError(
                "mask_feature_source='merged' requires multiscale_merge to produce a "
                "merged feature map. Use multiscale_merge='conv' or 'deformable', or "
                "switch mask_feature_source to 'highest_resolution'."
            )

        self.channel_aggregation = channel_aggregation
        self.mask_feature_source = mask_feature_source
        self.mask_output_dim = int(mask_output_dim)

        self.decoder = ImageTextFusion(
            num_layers=num_decode_layers,
            embed_dim=decode_embed_dim,
            image_dim=image_dim,
            text_dim=text_dim,
            nhead=nhead,
            ffn_dim=decode_ffn_dim,
            dropout=dropout,
            activation=activation,
            num_feature_levels=num_feature_levels,
            pos_encode=decode_pos_encode,
            image_self_attention=image_self_attention,
            window_size=decode_window_size,
            use_shifted_windows=use_shifted_windows,
            multiscale_merge=multiscale_merge,
            deformable_num_points=deformable_num_points,
            deformable_offset_scale=deformable_offset_scale,
            conv_merge_refinement=conv_merge_refinement,
        )
        self.channel_aggregator = None
        if self.channel_aggregation != "none":
            self.channel_aggregator = ChannelAggregator(
                image_dim=decode_embed_dim,
                text_dim=text_dim,
                hidden_dim=aggregator_dim,
                output_dim=decode_embed_dim,
                dropout=dropout,
                activation=activation,
            )
        self.mask_head = MaskHead(
            image_dim=decode_embed_dim,
            text_dim=text_dim,
            hidden_dim=(
                mask_head_hidden_dim
                if mask_head_hidden_dim is not None
                else aggregator_dim
            ),
            output_dim=mask_output_dim,
            upsample_scales=mask_upsample_scales,
            dropout=dropout,
            activation=activation,
            convolution=mask_convolution,
        )
        self.box_projection = nn.Conv2d(1, decode_embed_dim, kernel_size=1, bias=False)
        nn.init.zeros_(self.box_projection.weight)

    @staticmethod
    def _rasterize_boxes(
        boxes: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                f"Expected normalized boxes shaped [B,4], got {tuple(boxes.shape)}."
            )
        y = (
            torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5
        ) / height
        x = (torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5) / width
        inside_y = (y[None, :, None] >= boxes[:, 1, None, None]) & (
            y[None, :, None] <= boxes[:, 3, None, None]
        )
        inside_x = (x[None, None, :] >= boxes[:, 0, None, None]) & (
            x[None, None, :] <= boxes[:, 2, None, None]
        )
        return (inside_y & inside_x).to(boxes.dtype).unsqueeze(1)

    def _select_mask_features(
        self,
        fused_feature_list: list[torch.Tensor],
        merged_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.mask_feature_source == "highest_resolution":
            return fused_feature_list[0]
        if merged_features is None:
            raise RuntimeError(
                "Expected merged decoder features, but the decoder returned None."
            )
        return merged_features

    def _apply_channel_aggregation(
        self,
        image_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.channel_aggregator is None:
            return image_features, None, None

        channel_logits = self.channel_aggregator(
            image_features,
            text_embeddings,
            text_padding_mask=text_padding_mask,
        )
        if self.channel_aggregation == "sigmoid":
            # A zero logit is the identity rather than suppressing every channel
            # by one half at initialization.
            channel_modulation = 2.0 * ChannelAggregator.compute_gates(channel_logits)
        else:
            channel_modulation = ChannelAggregator.compute_weights(
                channel_logits, dim=-1
            )
            channel_modulation = channel_modulation * channel_modulation.shape[-1]

        modulated_features = image_features * channel_modulation.unsqueeze(
            -1
        ).unsqueeze(-1)
        return modulated_features, channel_logits, channel_modulation

    def forward(
        self,
        multi_image_features: Iterable[torch.Tensor],
        text_embeddings: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
        *,
        object_boxes: torch.Tensor | None = None,
        mask_output_size: tuple[int, int] | None = None,
        return_intermediates: bool = True,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        fused_feature_list, merged_features = self.decoder(
            multi_image_features,
            text_embeddings,
            text_padding_mask=text_padding_mask,
            return_merged_features=True,
        )

        mask_features = self._select_mask_features(fused_feature_list, merged_features)
        if object_boxes is not None:
            box_masks = self._rasterize_boxes(
                object_boxes,
                height=mask_features.shape[-2],
                width=mask_features.shape[-1],
            )
            mask_features = mask_features + self.box_projection(box_masks)
        modulated_features, channel_logits, channel_modulation = (
            self._apply_channel_aggregation(
                mask_features,
                text_embeddings,
                text_padding_mask=text_padding_mask,
            )
        )
        mask_logits = self.mask_head(
            modulated_features,
            text_embeddings,
            text_padding_mask=text_padding_mask,
            output_size=mask_output_size,
        )

        if self.mask_output_dim == 1:
            mask_logits = mask_logits.squeeze(1)

        outputs: dict[str, torch.Tensor | list[torch.Tensor] | None] = {
            "mask_logits": mask_logits,
        }
        if not return_intermediates:
            return outputs
        outputs.update(
            {
                "fused_feature_list": fused_feature_list,
                "merged_features": merged_features,
                "mask_features": mask_features,
                "modulated_features": modulated_features,
                "channel_logits": channel_logits,
                "channel_modulation": channel_modulation,
            }
        )
        return outputs
