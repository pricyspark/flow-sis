from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers import RTDetrImageProcessor

from flowsis.pretrained import (
    BaseDetector,
    resolve_detector,
)
from flowsis.pretrained.dfine import DFineDetector
from flowsis.pretrained.rtdetrv2 import RTDetrV2Detector
from flowsis.pretrained.image_processing import (
    image_to_rgb_tensor,
    preprocess_detr_bgr_frame,
    preprocess_detr_images,
)


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

    def save_pretrained(self, output_dir: Path) -> None:
        (output_dir / "model.txt").write_text("fake\n")


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

    def save_pretrained(self, output_dir: Path) -> None:
        (output_dir / "processor.txt").write_text("fake\n")


class FakeDetector(BaseDetector):
    architecture = "rtdetrv2"
    expected_model_types = ("rt_detr_v2",)


def build_detector(*, image_size: int = 8) -> FakeDetector:
    return FakeDetector(
        FakeProcessor(),
        FakeModel(),
        source="fake",
        image_size=image_size,
    )


def test_backend_adapters_are_siblings() -> None:
    assert not issubclass(DFineDetector, RTDetrV2Detector)
    assert issubclass(DFineDetector, BaseDetector)
    assert issubclass(RTDetrV2Detector, BaseDetector)


def test_single_tensor_is_one_image() -> None:
    detector = build_detector()
    detector.preprocess(torch.zeros(3, 8, 8))
    assert detector._processor.batch_size == 1


def test_inference_contract_and_training_mode_restoration() -> None:
    detector = build_detector()
    detector.train()
    result = detector.infer(np.zeros((8, 8, 3), dtype=np.uint8))
    assert detector.training
    assert len(result.detections) == 1
    assert [tuple(feature.shape) for feature in result.feature_maps] == [
        (1, 4, 2, 2),
        (1, 4, 1, 1),
    ]


def test_detector_image_size_is_instance_configuration() -> None:
    detector = build_detector(image_size=24)

    batch = detector.preprocess(torch.zeros(3, 8, 8))

    assert detector.image_size == 24
    assert batch["pixel_values"].shape[-2:] == (24, 24)


def test_detector_rejects_invalid_image_size() -> None:
    with pytest.raises(ValueError, match="image_size must be positive"):
        build_detector(image_size=0)


def test_detector_checkpoint_records_image_size(tmp_path: Path) -> None:
    build_detector(image_size=24).save_pretrained(tmp_path)

    metadata = json.loads((tmp_path / "flowsis_detector.json").read_text())

    assert metadata["image_size"] == 24


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


def test_pil_conversion_returns_writable_tensor_without_warning() -> None:
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tensor = image_to_rgb_tensor(image)

    tensor[0, 0, 0] = 1
    assert not caught


def test_device_batch_preprocessing_matches_reference_images_and_labels() -> None:
    processor = RTDetrImageProcessor()
    image = np.random.default_rng(11).integers(
        0,
        256,
        size=(31, 47, 3),
        dtype=np.uint8,
    )
    annotations = [
        {
            "image_id": 9,
            "annotations": [
                {
                    "bbox": [4.0, 5.0, 12.0, 10.0],
                    "category_id": 2,
                    "area": 120.0,
                    "iscrowd": 0,
                }
            ],
        }
    ]
    reference = processor.preprocess(
        images=[image],
        annotations=annotations,
        return_tensors="pt",
        size={"shortest_edge": 24, "longest_edge": 24},
        do_pad=True,
        pad_size={"height": 24, "width": 24},
    )

    pixels, mask, labels = preprocess_detr_images(
        processor,
        torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0),
        image_size=24,
        device=torch.device("cpu"),
        annotations=annotations,
    )

    assert labels is not None
    assert torch.equal(mask, reference["pixel_mask"])
    assert torch.allclose(
        pixels,
        reference["pixel_values"],
        atol=float(processor.rescale_factor) + 1e-7,
        rtol=0.0,
    )
    for key in ("size", "image_id", "class_labels", "iscrowd", "orig_size"):
        assert torch.equal(labels[0][key], reference["labels"][0][key])
    for key in ("boxes", "area"):
        assert torch.allclose(labels[0][key], reference["labels"][0][key])
