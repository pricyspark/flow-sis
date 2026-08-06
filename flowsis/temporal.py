from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TemporalOutput:
    """Outputs and diagnostics produced for one frame pair."""

    final_logits: torch.Tensor
    propagated_logits: torch.Tensor
    propagation_gate: torch.Tensor
    residual_gate: torch.Tensor
    logit_residual: torch.Tensor
    warp_validity: torch.Tensor
    photometric_residual: torch.Tensor


@dataclass(frozen=True)
class TemporalState:
    """Detached recurrent state retained between inference frames."""

    frame: torch.Tensor
    logits: torch.Tensor
    identity: int | str | None = None

    @classmethod
    def from_output(
        cls,
        frame: torch.Tensor,
        output: TemporalOutput,
        *,
        identity: int | str | None = None,
    ) -> TemporalState:
        return cls(
            frame=frame.detach(),
            logits=output.final_logits.detach(),
            identity=identity,
        )


def warp_backward(
    values: torch.Tensor,
    backward_flow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``values`` using current-to-previous flow.

    ``backward_flow[:, :, y, x]`` contains the pixel displacement from a
    location in the output/current frame to its source in ``values``. The
    returned validity tensor is one where the source pixel center lies inside
    the input image.
    """
    if values.ndim != 4:
        raise ValueError(
            "values must have shape [B, C, H, W], "
            f"but received {tuple(values.shape)}."
        )
    if backward_flow.ndim != 4 or backward_flow.shape[1] != 2:
        raise ValueError(
            "backward_flow must have shape [B, 2, H, W], "
            f"but received {tuple(backward_flow.shape)}."
        )
    if values.shape[0] != backward_flow.shape[0]:
        raise ValueError("values and backward_flow must have the same batch size.")
    if values.shape[-2:] != backward_flow.shape[-2:]:
        raise ValueError("values and backward_flow must have the same spatial size.")
    if values.device != backward_flow.device:
        raise ValueError("values and backward_flow must be on the same device.")

    batch_size, _, height, width = values.shape
    coordinate_dtype = backward_flow.dtype
    y, x = torch.meshgrid(
        torch.arange(height, device=values.device, dtype=coordinate_dtype),
        torch.arange(width, device=values.device, dtype=coordinate_dtype),
        indexing="ij",
    )
    source_x = x.unsqueeze(0) + backward_flow[:, 0]
    source_y = y.unsqueeze(0) + backward_flow[:, 1]

    grid_x = (2.0 * (source_x + 0.5) / width) - 1.0
    grid_y = (2.0 * (source_y + 0.5) / height) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).to(values.dtype)

    warped = F.grid_sample(
        values,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    validity = (
        (source_x >= 0.0)
        & (source_x <= width - 1)
        & (source_y >= 0.0)
        & (source_y <= height - 1)
    ).unsqueeze(1)
    return warped, validity.to(values.dtype)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class TemporalRefinementBranch(nn.Module):
    """Flow-guided corrective refinement for binary segmentation logits.

    Frames must be floating-point RGB tensors with the same normalization.
    ``backward_flow`` is expressed in pixels and maps current-frame positions
    into the previous frame. Gradient policy is controlled by the caller, so
    segmentation gradients can reach a trainable flow estimator.
    """

    _INPUT_CHANNELS = 14

    def __init__(
        self,
        *,
        channels: tuple[int, ...] = (32, 64, 96),
        residual_limit: float = 5.0,
        propagation_gate_bias: float = -3.0,
        residual_gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain positive integers.")
        if residual_limit <= 0:
            raise ValueError("residual_limit must be positive.")

        self.residual_limit = float(residual_limit)

        encoder: list[nn.Module] = []
        in_channels = self._INPUT_CHANNELS
        for index, out_channels in enumerate(channels):
            encoder.append(
                _ConvBlock(
                    in_channels,
                    out_channels,
                    stride=1 if index == 0 else 2,
                )
            )
            in_channels = out_channels
        self.encoder = nn.ModuleList(encoder)

        decoder: list[nn.Module] = []
        for deep_channels, skip_channels in zip(
            reversed(channels[1:]),
            reversed(channels[:-1]),
            strict=True,
        ):
            decoder.append(_ConvBlock(deep_channels + skip_channels, skip_channels))
        self.decoder = nn.ModuleList(decoder)

        self.gate_head = nn.Conv2d(channels[0], 2, kernel_size=1)
        self.residual_head = nn.Conv2d(channels[0], 1, kernel_size=1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias[0], propagation_gate_bias)
        nn.init.constant_(self.gate_head.bias[1], residual_gate_bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @staticmethod
    def _validate_inputs(
        current_frame: torch.Tensor,
        previous_frame: torch.Tensor,
        current_base_logits: torch.Tensor,
        previous_logits: torch.Tensor,
        backward_flow: torch.Tensor,
    ) -> None:
        if current_frame.ndim != 4 or current_frame.shape[1] != 3:
            raise ValueError("current_frame must have shape [B, 3, H, W].")
        if previous_frame.shape != current_frame.shape:
            raise ValueError(
                "previous_frame must have the same shape as current_frame."
            )
        expected_logit_shape = (current_frame.shape[0], 1, *current_frame.shape[-2:])
        if current_base_logits.shape != expected_logit_shape:
            raise ValueError(
                "current_base_logits must have shape "
                f"{expected_logit_shape}, but received "
                f"{tuple(current_base_logits.shape)}."
            )
        if previous_logits.shape != expected_logit_shape:
            raise ValueError(
                "previous_logits must have shape "
                f"{expected_logit_shape}, but received {tuple(previous_logits.shape)}."
            )
        expected_flow_shape = (current_frame.shape[0], 2, *current_frame.shape[-2:])
        if backward_flow.shape != expected_flow_shape:
            raise ValueError(
                f"backward_flow must have shape {expected_flow_shape}, "
                f"but received {tuple(backward_flow.shape)}."
            )
        tensors = (
            current_frame,
            previous_frame,
            current_base_logits,
            previous_logits,
            backward_flow,
        )
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise ValueError("All temporal inputs must be floating-point tensors.")
        if any(tensor.device != current_frame.device for tensor in tensors[1:]):
            raise ValueError("All temporal inputs must be on the same device.")

    @staticmethod
    def _reliability_features(
        current_frame: torch.Tensor,
        warped_previous_frame: torch.Tensor,
        current_base_logits: torch.Tensor,
        propagated_logits: torch.Tensor,
        backward_flow: torch.Tensor,
        validity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = current_frame.shape[-2:]
        flow_scale = backward_flow.new_tensor((width, height)).view(1, 2, 1, 1)
        normalized_flow = backward_flow / flow_scale
        rgb_residual = (current_frame - warped_previous_frame).abs()
        photometric_residual = rgb_residual.mean(dim=1, keepdim=True)
        logit_difference = propagated_logits - current_base_logits

        features = torch.cat(
            (
                current_frame,
                warped_previous_frame,
                photometric_residual,
                normalized_flow,
                validity,
                torch.tanh(current_base_logits / 5.0),
                torch.tanh(propagated_logits / 5.0),
                torch.tanh(logit_difference / 5.0),
                torch.tanh(logit_difference.abs() / 5.0),
            ),
            dim=1,
        )
        return features, photometric_residual

    def forward(
        self,
        current_frame: torch.Tensor,
        previous_frame: torch.Tensor,
        current_base_logits: torch.Tensor,
        previous_logits: torch.Tensor,
        backward_flow: torch.Tensor,
    ) -> TemporalOutput:
        self._validate_inputs(
            current_frame,
            previous_frame,
            current_base_logits,
            previous_logits,
            backward_flow,
        )
        flow = backward_flow
        propagated_logits, validity = warp_backward(previous_logits, flow)
        warped_previous_frame, _ = warp_backward(previous_frame, flow)
        inputs, photometric_residual = self._reliability_features(
            current_frame,
            warped_previous_frame,
            current_base_logits,
            propagated_logits,
            flow,
            validity,
        )

        skips: list[torch.Tensor] = []
        features = inputs
        for block in self.encoder:
            features = block(features)
            skips.append(features)
        for block, skip in zip(self.decoder, reversed(skips[:-1]), strict=True):
            features = F.interpolate(
                features,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            features = block(torch.cat((features, skip), dim=1))

        gate_logits = self.gate_head(features)
        propagation_gate = gate_logits[:, :1].sigmoid() * validity
        residual_gate = gate_logits[:, 1:].sigmoid()
        logit_residual = self.residual_limit * torch.tanh(self.residual_head(features))

        blended_logits = current_base_logits + propagation_gate * (
            propagated_logits - current_base_logits
        )
        final_logits = blended_logits + residual_gate * logit_residual
        return TemporalOutput(
            final_logits=final_logits,
            propagated_logits=propagated_logits,
            propagation_gate=propagation_gate,
            residual_gate=residual_gate,
            logit_residual=logit_residual,
            warp_validity=validity,
            photometric_residual=photometric_residual,
        )


__all__ = [
    "TemporalOutput",
    "TemporalRefinementBranch",
    "TemporalState",
    "warp_backward",
]
