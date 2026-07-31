from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from numpy.typing import NDArray

from flowsis.data import LabelPrompts
from flowsis.head_checkpoint import load_head
from flowsis.pretrained import DetectorArchitecture, load_detector
from flowsis.selection import SelectionResult, normalize_box, select_detection
from flowsis.utils import build_autocast_context, get_device


class FlowSISBase(nn.Module):
    """Stateful detector-selection and mask-inference pipeline."""

    def __init__(
        self,
        detector_name_or_path: str,
        head_path: str | Path,
        text_embeddings_dir: str | Path,
        *,
        detector_architecture: DetectorArchitecture | None = None,
        detection_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        history_size: int = 12,
        image_size: int = 640,
        device: str | torch.device | None = None,
        use_amp: bool = True,
        device_preprocess: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= detection_threshold <= 1.0:
            raise ValueError("Detection threshold must be between zero and one.")
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("Mask threshold must be between zero and one.")
        if history_size <= 0:
            raise ValueError("History size must be positive.")
        if image_size <= 0:
            raise ValueError("Image size must be positive.")

        self.detector = load_detector(
            detector_name_or_path,
            architecture=detector_architecture,
            image_size=image_size,
        )
        self.head, self.head_checkpoint_path = load_head(head_path)
        self.text_embeddings_dir = Path(text_embeddings_dir)
        self.id2label = self.detector.label_names
        self.prompt_cache: dict[str, torch.Tensor] = {}
        self.history: deque[Mapping[str, Any]] = deque(maxlen=history_size)
        self.previous_selection: SelectionResult | None = None
        self.current_selection: SelectionResult | None = None
        self.detection_threshold = detection_threshold
        self.mask_threshold = mask_threshold
        self.use_amp = use_amp
        self.device_preprocess = device_preprocess

        selected_device = get_device() if device is None else torch.device(device)
        self.to(selected_device)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.head.parameters()).device

    @property
    def selected_label(self) -> str | None:
        if self.current_selection is None:
            return None
        label = self.current_selection.label
        return self.id2label.get(label, f"class_{label}")

    def reset(self) -> None:
        """Clear temporal selection state before processing a new stream."""
        self.history.clear()
        self.previous_selection = None
        self.current_selection = None

    @torch.inference_mode()
    def predict_logits(
        self,
        feature_maps: list[torch.Tensor],
        text_embeddings: torch.Tensor,
        box: torch.Tensor,
        *,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Predict one selected object's mask logits as a device-resident [H,W] tensor."""
        with build_autocast_context(enabled=self.use_amp, device=self.device):
            output = self.head(
                feature_maps,
                text_embeddings.unsqueeze(0),
                object_boxes=box,
                mask_output_size=output_size,
                return_intermediates=False,
            )
        mask_logits = cast(torch.Tensor, output["mask_logits"])
        if mask_logits.ndim != 3 or mask_logits.shape[0] != 1:
            raise RuntimeError(
                "FlowSISBase expects one binary mask shaped [1,H,W], "
                f"but received {tuple(mask_logits.shape)}."
            )
        return mask_logits[0]

    def binarize_logits(
        self,
        mask_logits: torch.Tensor,
        *,
        threshold: float | None = None,
    ) -> torch.Tensor:
        """Convert mask logits to a boolean tensor without changing its device."""
        resolved_threshold = self.mask_threshold if threshold is None else threshold
        if not 0.0 <= resolved_threshold <= 1.0:
            raise ValueError("Mask threshold must be between zero and one.")
        return mask_logits.sigmoid() >= resolved_threshold

    @torch.inference_mode()
    def predict_mask(
        self,
        feature_maps: list[torch.Tensor],
        text_embeddings: torch.Tensor,
        box: torch.Tensor,
        *,
        output_size: tuple[int, int],
        mask_threshold: float | None = None,
    ) -> torch.Tensor:
        """Predict and binarize one mask, leaving the result on the model device."""
        logits = self.predict_logits(
            feature_maps,
            text_embeddings,
            box,
            output_size=output_size,
        )
        return self.binarize_logits(logits, threshold=mask_threshold)

    @torch.inference_mode()
    def infer_logits(self, square_bgr: NDArray) -> torch.Tensor | None:
        """Process one BGR frame and return selected mask logits on the model device."""
        if square_bgr.ndim != 3 or square_bgr.shape[2] != 3:
            raise ValueError(f"Expected an HWC BGR frame, got {square_bgr.shape}.")
        height, width = square_bgr.shape[:2]

        with build_autocast_context(enabled=self.use_amp, device=self.device):
            inference = self.detector.infer_frame(
                square_bgr,
                threshold=self.detection_threshold,
                device_preprocess=self.device_preprocess,
            )

        detections = {
            key: value.detach().cpu().tolist()
            for key, value in inference.detections[0].items()
        }
        self.history.append(detections)
        selection = select_detection(self.history, self.previous_selection)
        self.current_selection = selection
        if selection is None:
            return None

        selected_label = cast(str, self.selected_label)
        text_embeddings = LabelPrompts.load_embeddings(
            selected_label,
            self.text_embeddings_dir,
            self.prompt_cache,
            device=self.device,
        )
        box = torch.tensor(
            [normalize_box(selection, width=width, height=height)],
            device=self.device,
            dtype=torch.float32,
        )
        logits = self.predict_logits(
            list(inference.feature_maps),
            text_embeddings,
            box,
            output_size=(height, width),
        )
        self.previous_selection = selection
        return logits

    @torch.inference_mode()
    def infer(
        self,
        square_bgr: NDArray,
        *,
        mask_threshold: float | None = None,
    ) -> torch.Tensor | None:
        """Process one BGR frame and return a boolean mask on the model device."""
        logits = self.infer_logits(square_bgr)
        if logits is None:
            return None
        return self.binarize_logits(logits, threshold=mask_threshold)
