from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from .base import FlowSISBase
from .temporal import TemporalOutput, TemporalRefinementBranch, TemporalState

Box = tuple[float, float, float, float]
Identity = int | str | None


def _box_iou(first: Box, second: Box) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def bgr_frame_to_rgb_tensor(
    frame_bgr: NDArray,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Convert one HWC BGR frame in the 0-255 range to BCHW RGB in [0, 1]."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected an HWC BGR frame, got {frame_bgr.shape}.")
    frame_rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
    return (
        torch.from_numpy(frame_rgb)
        .to(device=device, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .div(255.0)
    )


class FlowSIS(nn.Module):
    """Compose base segmentation, optical flow, and recurrent refinement."""

    def __init__(
        self,
        base: nn.Module,
        flow_estimator: nn.Module,
        temporal: TemporalRefinementBranch | None = None,
        *,
        minimum_selection_iou: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0.0 <= minimum_selection_iou <= 1.0:
            raise ValueError("minimum_selection_iou must be between zero and one.")
        self.base = cast(FlowSISBase, base)
        self.flow_estimator = flow_estimator
        self.temporal = temporal or TemporalRefinementBranch()
        self.minimum_selection_iou = minimum_selection_iou
        self.temporal_state: TemporalState | None = None
        self.previous_selection_box: Box | None = None
        self.last_temporal_output: TemporalOutput | None = None

    def reset(self) -> None:
        """Clear base selection and recurrent segmentation state."""
        self.base.reset()
        self._reset_temporal_state()

    def _reset_temporal_state(self) -> None:
        self.temporal_state = None
        self.previous_selection_box = None
        self.last_temporal_output = None

    def forward(
        self,
        current_frame: torch.Tensor,
        previous_frame: torch.Tensor,
        current_base_logits: torch.Tensor,
        previous_logits: torch.Tensor,
        backward_flow: torch.Tensor | None = None,
    ) -> TemporalOutput:
        """Refine one tensor frame pair, estimating flow when it is not supplied."""
        if backward_flow is None:
            backward_flow = self.flow_estimator(current_frame, previous_frame)
        return self.temporal(
            current_frame,
            previous_frame,
            current_base_logits,
            previous_logits,
            backward_flow,
        )

    def _selection_identity(self, identity: Identity) -> Identity:
        if identity is not None:
            return identity
        selection = self.base.current_selection
        return None if selection is None else selection.label

    def _selection_box(self) -> Box | None:
        selection = self.base.current_selection
        return None if selection is None else selection.box

    def _state_is_compatible(
        self,
        frame: torch.Tensor,
        identity: Identity,
        selection_box: Box | None,
    ) -> bool:
        state = self.temporal_state
        if (
            state is None
            or state.frame.shape != frame.shape
            or state.identity != identity
        ):
            return False
        if selection_box is None or self.previous_selection_box is None:
            return True
        return (
            _box_iou(selection_box, self.previous_selection_box)
            >= self.minimum_selection_iou
        )

    def _store_state(
        self,
        frame: torch.Tensor,
        logits: torch.Tensor,
        identity: Identity,
        selection_box: Box | None,
    ) -> None:
        self.temporal_state = TemporalState(
            frame=frame.detach(),
            logits=logits.detach(),
            identity=identity,
        )
        self.previous_selection_box = selection_box

    @torch.inference_mode()
    def infer_logits(
        self,
        frame_bgr: NDArray,
        *,
        identity: int | str | None = None,
    ) -> torch.Tensor | None:
        """Process one BGR frame and return final binary logits as ``[H, W]``."""
        base_logits = self.base.infer_logits(frame_bgr)
        if base_logits is None:
            self._reset_temporal_state()
            return None

        frame = bgr_frame_to_rgb_tensor(frame_bgr, device=base_logits.device)
        batched_base_logits = base_logits.unsqueeze(0).unsqueeze(0)
        resolved_identity = self._selection_identity(identity)
        selection_box = self._selection_box()
        if not self._state_is_compatible(frame, resolved_identity, selection_box):
            self.last_temporal_output = None
            self._store_state(
                frame,
                batched_base_logits,
                resolved_identity,
                selection_box,
            )
            return base_logits

        state = self.temporal_state
        assert state is not None
        output = self(
            frame,
            state.frame,
            batched_base_logits,
            state.logits,
        )
        self.last_temporal_output = output
        self._store_state(
            frame,
            output.final_logits,
            resolved_identity,
            selection_box,
        )
        return output.final_logits[0, 0]

    @torch.inference_mode()
    def infer(
        self,
        frame_bgr: NDArray,
        *,
        identity: int | str | None = None,
        mask_threshold: float | None = None,
    ) -> torch.Tensor | None:
        """Process one BGR frame and return the final thresholded mask."""
        logits = self.infer_logits(frame_bgr, identity=identity)
        if logits is None:
            return None
        return self.base.binarize_logits(logits, threshold=mask_threshold)


__all__ = ["FlowSIS", "bgr_frame_to_rgb_tensor"]
