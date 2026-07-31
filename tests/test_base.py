from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

import flowsis.base as base_module
from flowsis.base import FlowSISBase


class FakeDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.label_names = {1: "object"}
        self.has_detection = True

    def infer_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        threshold: float,
        device_preprocess: bool,
    ) -> Any:
        if self.has_detection:
            detections = {
                "boxes": self.anchor.new_tensor([[1.0, 1.0, 7.0, 7.0]]),
                "scores": self.anchor.new_tensor([0.9]),
                "labels": torch.tensor([1], device=self.anchor.device),
            }
        else:
            detections = {
                "boxes": self.anchor.new_empty((0, 4)),
                "scores": self.anchor.new_empty((0,)),
                "labels": torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.anchor.device,
                ),
            }
        return SimpleNamespace(
            detections=[detections],
            feature_maps=(self.anchor.new_ones((1, 4, 2, 2)),),
        )


class FakeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        feature_maps: list[torch.Tensor],
        text_embeddings: torch.Tensor,
        *,
        object_boxes: torch.Tensor,
        mask_output_size: tuple[int, int],
        return_intermediates: bool,
    ) -> dict[str, torch.Tensor]:
        batch_size = feature_maps[0].shape[0]
        logits = self.anchor.expand(batch_size, *mask_output_size)
        return {"mask_logits": logits}


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FlowSISBase:
    detector = FakeDetector()
    head = FakeHead()
    monkeypatch.setattr(base_module, "load_detector", lambda *args, **kwargs: detector)
    monkeypatch.setattr(base_module, "load_head", lambda *args, **kwargs: (head, tmp_path))
    monkeypatch.setattr(
        base_module.LabelPrompts,
        "load_embeddings",
        lambda *args, device, **kwargs: torch.ones(2, 4, device=device),
    )
    return FlowSISBase(
        "detector",
        tmp_path,
        tmp_path,
        device="cpu",
        use_amp=True,
    )


def test_infer_logits_and_binarization_remain_on_model_device(
    model: FlowSISBase,
) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    logits = model.infer_logits(frame)

    assert logits is not None
    assert logits.shape == (8, 8)
    assert logits.device == model.device
    mask = model.binarize_logits(logits)
    assert mask.dtype == torch.bool
    assert mask.device == model.device
    assert mask.all()
    assert model.previous_selection is not None
    assert model.current_selection is model.previous_selection
    assert model.selected_label == "object"
    assert len(model.history) == 1


def test_infer_returns_none_without_detection_and_reset_clears_state(
    model: FlowSISBase,
) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert model.infer(frame) is not None

    model.reset()
    model.detector.has_detection = False

    assert model.infer(frame) is None
    assert model.previous_selection is None
    assert model.current_selection is None
    assert model.selected_label is None
    assert len(model.history) == 1


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("detection_threshold", -0.1),
        ("mask_threshold", 1.1),
        ("history_size", 0),
        ("image_size", 0),
    ],
)
def test_invalid_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argument: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        FlowSISBase(
            "detector",
            tmp_path,
            tmp_path,
            device="cpu",
            **{argument: value},
        )
