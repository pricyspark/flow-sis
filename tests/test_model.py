from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from flowsis.model import FlowSIS, bgr_frame_to_rgb_tensor
from flowsis.temporal import TemporalRefinementBranch


class FakeBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.current_selection = None
        self.has_detection = True
        self.box = (1.0, 1.0, 7.0, 7.0)
        self.label = 1
        self.reset_count = 0

    def infer_logits(self, frame_bgr: np.ndarray) -> torch.Tensor | None:
        if not self.has_detection:
            self.current_selection = None
            return None
        self.current_selection = SimpleNamespace(label=self.label, box=self.box)
        height, width = frame_bgr.shape[:2]
        return self.anchor.expand(height, width)

    def binarize_logits(
        self,
        logits: torch.Tensor,
        *,
        threshold: float | None = None,
    ) -> torch.Tensor:
        resolved_threshold = 0.5 if threshold is None else threshold
        return logits.sigmoid() >= resolved_threshold

    def reset(self) -> None:
        self.current_selection = None
        self.reset_count += 1


class FakeFlowEstimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.displacement = nn.Parameter(torch.tensor(0.0))
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        current_frame: torch.Tensor,
        previous_frame: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append(
            (current_frame.detach().clone(), previous_frame.detach().clone())
        )
        batch_size, _, height, width = current_frame.shape
        return self.displacement.expand(batch_size, 2, height, width)


def make_model() -> tuple[FlowSIS, FakeBase, FakeFlowEstimator]:
    base = FakeBase()
    flow = FakeFlowEstimator()
    model = FlowSIS(
        base,
        flow,
        TemporalRefinementBranch(channels=(8,)),
    )
    return model, base, flow


def test_frame_conversion_changes_bgr_to_normalized_rgb() -> None:
    frame = np.array([[[10, 20, 30]]], dtype=np.uint8)

    tensor = bgr_frame_to_rgb_tensor(frame, device=torch.device("cpu"))

    torch.testing.assert_close(
        tensor[0, :, 0, 0],
        torch.tensor([30.0, 20.0, 10.0]) / 255.0,
    )


def test_inference_anchors_first_frame_then_uses_flow_and_final_state() -> None:
    model, _, flow = make_model()
    first_frame = np.full((8, 8, 3), (10, 20, 30), dtype=np.uint8)
    second_frame = np.full((8, 8, 3), (40, 50, 60), dtype=np.uint8)

    first_logits = model.infer_logits(first_frame)
    second_logits = model.infer_logits(second_frame)

    assert first_logits is not None
    assert second_logits is not None
    assert len(flow.calls) == 1
    current, previous = flow.calls[0]
    torch.testing.assert_close(current[0, :, 0, 0], torch.tensor([60, 50, 40]) / 255)
    torch.testing.assert_close(previous[0, :, 0, 0], torch.tensor([30, 20, 10]) / 255)
    assert model.last_temporal_output is not None
    assert model.temporal_state is not None
    torch.testing.assert_close(
        model.temporal_state.logits,
        model.last_temporal_output.final_logits,
    )


def test_identity_change_and_disjoint_selection_restart_temporal_state() -> None:
    model, base, flow = make_model()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    model.infer_logits(frame, identity="first")
    model.infer_logits(frame, identity="second")
    assert not flow.calls

    model.infer_logits(frame, identity="second")
    assert len(flow.calls) == 1

    base.box = (20.0, 20.0, 30.0, 30.0)
    model.infer_logits(frame, identity="second")
    assert len(flow.calls) == 1
    assert model.last_temporal_output is None


def test_missing_detection_and_reset_clear_state() -> None:
    model, base, _ = make_model()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert model.infer_logits(frame) is not None

    base.has_detection = False
    assert model.infer_logits(frame) is None
    assert model.temporal_state is None

    model.reset()
    assert base.reset_count == 1
    assert model.temporal_state is None


def test_tensor_forward_allows_segmentation_gradients_to_reach_flow() -> None:
    model, _, flow = make_model()
    current_frame = torch.rand(1, 3, 8, 8)
    previous_frame = torch.rand(1, 3, 8, 8)
    current_logits = torch.zeros(1, 1, 8, 8)
    previous_logits = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)

    output = model(
        current_frame,
        previous_frame,
        current_logits,
        previous_logits,
    )
    output.final_logits.square().mean().backward()

    assert flow.displacement.grad is not None
