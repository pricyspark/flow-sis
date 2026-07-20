from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
import torch.nn as nn
from numpy.typing import NDArray
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature

from .rtdetrv2 import RTDetrV2ForwardResult, RTDetrV2InferenceResult

DetectorForwardResult = RTDetrV2ForwardResult
DetectorInferenceResult = RTDetrV2InferenceResult
DetectorArchitecture = Literal["rtdetrv2", "dfine"]
DETECTOR_ARCHITECTURES: tuple[DetectorArchitecture, ...] = ("rtdetrv2", "dfine")
DetectorImage = Image.Image | NDArray | torch.Tensor
DetectorImages = DetectorImage | Iterable[DetectorImage]


class Detector(Protocol):
    """Interface used by FlowSIS detector training and mask-feature extraction."""

    processor: Any
    model: nn.Module

    @property
    def device(self) -> torch.device: ...

    def __call__(
        self,
        images: DetectorImages,
        annotations: Iterable[dict[str, Any]] | None = None,
        *,
        image_size: int | None = None,
        return_outputs: bool = False,
    ) -> DetectorForwardResult: ...

    def preprocess(
        self,
        images: DetectorImages,
        annotations: Iterable[dict[str, Any]] | None = None,
        *,
        image_size: int | None = 640,
    ) -> BatchFeature: ...

    def infer(
        self,
        images: DetectorImages,
        *,
        image_size: int | None = None,
        threshold: float = 0.1,
        flatten_outputs: bool = False,
    ) -> DetectorInferenceResult: ...

    def infer_preprocessed(
        self,
        pixel_values: torch.Tensor,
        *,
        original_sizes: list[tuple[int, int]],
        threshold: float = 0.1,
        pixel_mask: torch.Tensor | None = None,
        flatten_outputs: bool = False,
    ) -> DetectorInferenceResult: ...

    def train(self, mode: bool = True) -> Detector: ...
    def eval(self) -> Detector: ...
    def parameters(self, recurse: bool = True) -> Iterable[nn.Parameter]: ...
    def requires_grad_(self, requires_grad: bool = True) -> Detector: ...
    def save_pretrained(self, output_dir: str | Path) -> None: ...


def load_detector(
    architecture: DetectorArchitecture,
    model_name_or_path: str,
    *,
    cache_dir: str = "flowsis/models",
    num_labels: int | None = None,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    device: str | torch.device | None = None,
) -> Detector:
    kwargs = {
        "cache_dir": cache_dir,
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id,
        "device": device,
    }
    if architecture == "rtdetrv2":
        from .rtdetrv2 import RTDetrV2

        return RTDetrV2.from_pretrained(model_name_or_path, **kwargs)
    if architecture == "dfine":
        from .dfine import DFine

        return DFine.from_pretrained(model_name_or_path, **kwargs)
    raise ValueError(f"Unsupported detector architecture: {architecture!r}")


def extract_feature_maps(
    detector: Detector,
    images: Iterable[Image.Image],
    *,
    image_size: int,
) -> list[torch.Tensor]:
    """Run a detector and return its projected multiscale encoder maps."""
    batch = detector.preprocess(images, annotations=None, image_size=image_size)
    outputs = detector.model(
        pixel_values=batch["pixel_values"],
        pixel_mask=batch.get("pixel_mask"),
    )
    raw_feature_maps = getattr(outputs, "encoder_last_hidden_state", None)
    if not isinstance(raw_feature_maps, (list, tuple)) or not raw_feature_maps:
        raise RuntimeError(
            "Detector did not return a non-empty list of multiscale encoder feature maps."
        )
    feature_maps = list(raw_feature_maps)
    if any(
        not isinstance(feature, torch.Tensor) or feature.ndim != 4
        for feature in feature_maps
    ):
        shapes = [getattr(feature, "shape", None) for feature in feature_maps]
        raise RuntimeError(
            f"Expected detector feature maps shaped [B,C,H,W], got {shapes}."
        )
    return [feature.float() for feature in feature_maps]
