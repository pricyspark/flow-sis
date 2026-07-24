from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature
from transformers import RTDetrImageProcessor

from flowsis.pretrained import (
    BaseDetector,
    resolve_detector,
)
from flowsis.pretrained.dfine import DFineDetector
from flowsis.pretrained.rtdetrv2 import RTDetrV2Detector
from flowsis.pretrained.image_processing import preprocess_detr_bgr_frame


class FakeConfig:
    model_type = "rt_detr_v2"
    id2label = {0: "object"}
    num_labels = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "id2label": self.id2label,
            "num_labels": self.num_labels,
        }


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = FakeConfig()
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: torch.Tensor | None = None,
        labels: Any = None,
    ) -> SimpleNamespace:
        batch_size = pixel_values.shape[0]
        return SimpleNamespace(
            loss=pixel_values.mean() if labels is not None else None,
            loss_dict={"loss": pixel_values.mean()} if labels is not None else {},
            logits=torch.zeros(batch_size, 2, 1),
            encoder_last_hidden_state=[
                torch.ones(batch_size, 4, 2, 2),
                torch.ones(batch_size, 4, 1, 1),
            ],
        )


class FakeProcessor:
    do_rescale = True
    rescale_factor = 1.0 / 255.0
    do_normalize = False

    def __init__(self) -> None:
        self.batch_size = 0

    def preprocess(self, *, images: list[Any], **kwargs: Any) -> BatchFeature:
        self.batch_size = len(images)
        size = kwargs.get("pad_size", {"height": 8, "width": 8})
        batch = BatchFeature(
            {
                "pixel_values": torch.zeros(
                    len(images),
                    3,
                    size["height"],
                    size["width"],
                ),
                "pixel_mask": torch.ones(
                    len(images),
                    size["height"],
                    size["width"],
                    dtype=torch.bool,
                ),
            }
        )
        if "annotations" in kwargs:
            batch["labels"] = [{} for _ in images]
        return batch

    def post_process_object_detection(
        self,
        outputs: SimpleNamespace,
        *,
        threshold: float,
        target_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.empty(0, 4),
                "scores": torch.empty(0),
                "labels": torch.empty(0, dtype=torch.long),
            }
            for _ in target_sizes
        ]


class FakeDetector(BaseDetector):
    architecture = "rtdetrv2"
    expected_model_types = ("rt_detr_v2",)


def build_detector() -> FakeDetector:
    return FakeDetector(
        FakeProcessor(),
        FakeModel(),
        source="fake",
    )


def test_backend_adapters_are_siblings() -> None:
    assert not issubclass(DFineDetector, RTDetrV2Detector)
    assert issubclass(DFineDetector, BaseDetector)
    assert issubclass(RTDetrV2Detector, BaseDetector)


def test_single_tensor_is_one_image() -> None:
    detector = build_detector()
    detector.preprocess(torch.zeros(3, 8, 8), image_size=8)
    assert detector._processor.batch_size == 1


def test_inference_contract_and_training_mode_restoration() -> None:
    detector = build_detector()
    detector.train()
    result = detector.infer(np.zeros((8, 8, 3), dtype=np.uint8), image_size=8)
    assert detector.training
    assert len(result.detections) == 1
    assert [tuple(feature.shape) for feature in result.feature_maps] == [
        (1, 4, 2, 2),
        (1, 4, 1, 1),
    ]


def test_backbone_parameter_split_is_complete() -> None:
    detector = build_detector()
    backbone, other = detector.split_backbone_parameters()
    assert set(backbone) == set(detector._model.backbone.parameters())
    assert set(other) == set(detector._model.head.parameters())


def test_checkpoint_metadata_rejects_explicit_backend_mismatch(
    tmp_path: Path,
) -> None:
    (tmp_path / "flowsis_detector.json").write_text(
        '{"format_version": 1, "architecture": "rtdetrv2", '
        '"model_type": "rt_detr_v2"}\n'
    )
    with pytest.raises(ValueError, match="not the requested"):
        resolve_detector(tmp_path, architecture="dfine")


def test_device_preprocessing_matches_reference_processor_for_square_frame() -> None:
    processor = RTDetrImageProcessor()
    frame_bgr = np.random.default_rng(7).integers(
        0,
        256,
        size=(31, 31, 3),
        dtype=np.uint8,
    )
    reference = processor.preprocess(
        images=[frame_bgr[..., ::-1].copy()],
        return_tensors="pt",
        size={"shortest_edge": 24, "longest_edge": 24},
        do_pad=True,
        pad_size={"height": 24, "width": 24},
    )
    pixels, mask = preprocess_detr_bgr_frame(
        processor,
        frame_bgr,
        image_size=24,
        device=torch.device("cpu"),
    )
    assert torch.equal(mask, reference["pixel_mask"])
    assert torch.allclose(
        pixels,
        reference["pixel_values"],
        atol=float(processor.rescale_factor) + 1e-7,
        rtol=0.0,
    )
