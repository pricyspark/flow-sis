from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


class PTLFlowEstimator(nn.Module):
    """Adapt a PTLFlow model to a tensor-native current-to-previous interface.

    The public interface uses RGB float tensors in ``[B, 3, H, W]`` format.
    PTLFlow's model interface uses BGR image sequences in
    ``[B, N, 3, H, W]`` format, so the channel conversion is contained here.
    This adapter does not choose train/eval mode, gradient context, or parameter
    trainability.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def _validate_frames(
        current_frame: torch.Tensor,
        previous_frame: torch.Tensor,
    ) -> None:
        if current_frame.ndim != 4 or current_frame.shape[1] != 3:
            raise ValueError("current_frame must have shape [B, 3, H, W].")
        if previous_frame.shape != current_frame.shape:
            raise ValueError(
                "previous_frame must have the same shape as current_frame."
            )
        if (
            not current_frame.is_floating_point()
            or not previous_frame.is_floating_point()
        ):
            raise ValueError("Flow estimator inputs must be floating-point tensors.")
        if current_frame.device != previous_frame.device:
            raise ValueError("Flow estimator inputs must be on the same device.")

    def forward(
        self,
        current_frame: torch.Tensor,
        previous_frame: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate pixel flow from current-frame positions to the previous frame."""
        self._validate_frames(current_frame, previous_frame)
        rgb_images = torch.stack((current_frame, previous_frame), dim=1)
        bgr_images = rgb_images.flip(2)
        predictions = self.model({"images": bgr_images})
        if not isinstance(predictions, Mapping) or "flows" not in predictions:
            raise RuntimeError("PTLFlow model output must contain a 'flows' tensor.")

        flows = predictions["flows"]
        if not isinstance(flows, torch.Tensor):
            raise RuntimeError("PTLFlow 'flows' output must be a tensor.")
        if flows.ndim == 5:
            if flows.shape[1] != 1:
                raise RuntimeError(
                    "Expected one PTLFlow prediction for a two-frame input, "
                    f"but received shape {tuple(flows.shape)}."
                )
            flows = flows[:, 0]
        expected_shape = (current_frame.shape[0], 2, *current_frame.shape[-2:])
        if flows.shape != expected_shape:
            raise RuntimeError(
                f"Expected flow shape {expected_shape}, "
                f"but received {tuple(flows.shape)}."
            )
        return flows


def load_flow_estimator(
    model_name: str,
    checkpoint: str | None = None,
) -> PTLFlowEstimator:
    """Load a PTLFlow model without imposing a frozen or trainable policy."""
    import ptlflow

    model = ptlflow.get_model(model_name, ckpt_path=checkpoint)
    return PTLFlowEstimator(model)


__all__ = ["PTLFlowEstimator", "load_flow_estimator"]
