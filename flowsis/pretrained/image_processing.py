from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray


def preprocess_detr_bgr_frame(
    processor: Any,
    frame_bgr: NDArray,
    *,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the numerical transforms shared by the supported DETR processors."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected an HWC BGR frame, got {frame_bgr.shape}.")
    if frame_bgr.shape[0] != frame_bgr.shape[1]:
        raise ValueError(
            "Device-side DETR preprocessing requires a square frame so it remains "
            "numerically equivalent to the reference processor."
        )
    pixels = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(device)
    pixels = pixels.permute(2, 0, 1).flip(0).unsqueeze(0).float()
    pixels = F.interpolate(
        pixels,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    pixels.round_().clamp_(0.0, 255.0)
    if getattr(processor, "do_rescale", True):
        pixels.mul_(float(getattr(processor, "rescale_factor", 1.0 / 255.0)))
    if getattr(processor, "do_normalize", False):
        mean = torch.as_tensor(processor.image_mean, device=device).view(
            1, 3, 1, 1
        )
        std = torch.as_tensor(processor.image_std, device=device).view(1, 3, 1, 1)
        pixels.sub_(mean).div_(std)
    mask = torch.ones(
        (1, image_size, image_size),
        dtype=torch.bool,
        device=device,
    )
    return pixels, mask
