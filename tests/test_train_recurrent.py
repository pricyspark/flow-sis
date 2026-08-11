from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import get_scheduler

from flowsis.cli.train.train_recurrent import (
    SamplingConfig,
    TemporalSnippetDataset,
    anchored_snippet_indices,
    build_optimizer,
    compute_base_logits,
    compute_recurrent_objective,
    configure_trainability,
    load_temporal_checkpoint,
    sample_stride,
    save_temporal_checkpoint,
    train_one_epoch,
    validate_video_disjoint_splits,
)
from flowsis.temporal import TemporalOutput, TemporalRefinementBranch


class ZeroFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.ones(()))

    def forward(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        return current.new_zeros((current.shape[0], 2, *current.shape[-2:]))


class TinyTemporal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        current_frame: torch.Tensor,
        previous_frame: torch.Tensor,
        current_base_logits: torch.Tensor,
        previous_logits: torch.Tensor,
        backward_flow: torch.Tensor,
    ) -> TemporalOutput:
        final = current_base_logits + self.recurrent_weight * previous_logits
        zeros = torch.zeros_like(final)
        ones = torch.ones_like(final)
        return TemporalOutput(final, previous_logits, ones, ones, zeros, ones, zeros)


class TinyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.ones(()))

    @torch.inference_mode()
    def extract_feature_maps(
        self, frames: torch.Tensor, *, device_preprocess: bool
    ) -> tuple[torch.Tensor]:
        assert device_preprocess
        return (frames,)


class TinyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        features: tuple[torch.Tensor],
        text_embeddings: torch.Tensor,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        return {"mask_logits": features[0].mean(1) * self.scale}


def _inputs() -> tuple[torch.Tensor, ...]:
    frames = torch.rand(2, 4, 3, 5, 5)
    base_logits = torch.randn(2, 4, 1, 5, 5, requires_grad=True)
    targets = torch.randint(0, 2, (2, 4, 1, 5, 5)).float()
    supervised = torch.tensor([[False, False, True, True], [False, False, False, True]])
    return frames, base_logits, targets, supervised


def test_anchored_indices_and_weighted_stride_sampling() -> None:
    assert anchored_snippet_indices(20, 6, 3) == (5, 8, 11, 14, 17, 20)
    generator = torch.Generator().manual_seed(4)
    samples = [sample_stride((1, 2, 3), (0.6, 0.35, 0.05), generator=generator) for _ in range(5000)]
    frequencies = {stride: samples.count(stride) / len(samples) for stride in (1, 2, 3)}
    assert frequencies[1] == pytest.approx(0.60, abs=0.04)
    assert frequencies[2] == pytest.approx(0.35, abs=0.04)
    assert frequencies[3] == pytest.approx(0.05, abs=0.02)


def test_video_leakage_is_rejected() -> None:
    split = [{"objects": [{"video_id": 7}]}]
    with pytest.raises(ValueError, match="video-disjoint"):
        validate_video_disjoint_splits(split, split)


def test_sparse_supervision_is_normalized_per_snippet() -> None:
    frames, logits, targets, supervised = _inputs()
    result = compute_recurrent_objective(
        TinyTemporal(), ZeroFlow(), frames, logits, targets, supervised
    )
    # Repeating one sample's supervision must not change the other sample's weight.
    assert result["loss"].ndim == 0
    assert torch.isfinite(result["loss"])


def test_targets_are_not_teacher_forced_and_recurrence_keeps_gradients() -> None:
    frames, logits, targets, supervised = _inputs()
    temporal = TinyTemporal()
    first = compute_recurrent_objective(
        temporal, ZeroFlow(), frames, logits, targets, supervised
    )
    changed_targets = 1 - targets
    second = compute_recurrent_objective(
        temporal, ZeroFlow(), frames, logits, changed_targets, supervised
    )
    # Targets affect the loss, but both executions use the same recurrent graph.
    first["loss"].backward(retain_graph=True)
    assert temporal.recurrent_weight.grad is not None
    assert logits.grad is not None and bool((logits.grad[:, 0] != 0).any())
    assert not torch.allclose(first["loss"], second["loss"])


def test_frozen_modules_and_optional_base_head_gradients() -> None:
    detector, head, flow, temporal = TinyDetector(), TinyHead(), ZeroFlow(), TinyTemporal()
    configure_trainability(detector, head, flow, temporal, fine_tune_base_head=False)
    assert not any(parameter.requires_grad for parameter in detector.parameters())
    assert not any(parameter.requires_grad for parameter in head.parameters())
    assert not any(parameter.requires_grad for parameter in flow.parameters())
    configure_trainability(detector, head, flow, temporal, fine_tune_base_head=True)
    frames = torch.rand(1, 3, 3, 5, 5)
    logits = compute_base_logits(
        detector,
        head,
        frames,
        torch.rand(1, 2, 4),
        torch.rand(1, 3, 4),
        fine_tune_base_head=True,
    )
    targets = torch.ones(1, 3, 1, 5, 5)
    result = compute_recurrent_objective(
        temporal,
        flow,
        frames,
        logits,
        targets,
        torch.tensor([[False, False, True]]),
        base_loss_weight=1.0,
    )
    result["loss"].backward()
    assert head.scale.grad is not None
    assert all(parameter.grad is None for parameter in detector.parameters())
    assert all(parameter.grad is None for parameter in flow.parameters())


def test_checkpoint_round_trip_restores_training_state(tmp_path: Path) -> None:
    temporal = TemporalRefinementBranch(channels=(8,))
    head = TinyHead()
    optimizer = build_optimizer(temporal, head, lr=1e-3, base_head_lr=1e-4, weight_decay=0)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=3)
    sampling = SamplingConfig(3, (1, 2, 3), (0.6, 0.35, 0.05))
    path = save_temporal_checkpoint(
        tmp_path,
        temporal,
        optimizer,
        scheduler,
        epoch=2,
        global_step=5,
        scaler=None,
        detector_checkpoint="detector",
        base_head_checkpoint="head",
        flow_checkpoint="flow",
        image_size=8,
        sampling=sampling,
        base_head=head,
    )
    original = {name: value.clone() for name, value in temporal.state_dict().items()}
    temporal.requires_grad_(False)
    with torch.no_grad():
        for value in temporal.state_dict().values():
            value.zero_()
    epoch, step = load_temporal_checkpoint(
        path,
        temporal,
        optimizer=optimizer,
        scheduler=scheduler,
        base_head=head,
        expected_references={"detector": "detector", "base_head": "head", "flow": "flow"},
        image_size=8,
        sampling=sampling,
    )
    assert (epoch, step) == (2, 5)
    assert all(torch.equal(temporal.state_dict()[name], value) for name, value in original.items())


def test_snippet_dataset_marks_only_available_annotations(tmp_path: Path) -> None:
    mask = np.ones((6, 8), dtype=np.uint8)
    mask_path = tmp_path / "mask.npz"
    np.savez_compressed(mask_path, mask=mask)
    embedding_path = tmp_path / "embedding.pt"
    torch.save(torch.ones(2, 4), embedding_path)

    def record(frame_idx: int) -> dict[str, object]:
        return {
            "height": 6,
            "width": 8,
            "objects": [{
                "video_id": 1,
                "frame_idx": frame_idx,
                "category": 0,
                "bbox": [1, 1, 3, 3],
                "mask_path": str(mask_path),
                "text_embedding_path": str(embedding_path),
            }],
        }

    def frames(_: Path, indices: object) -> list[np.ndarray]:
        return [np.zeros((6, 8, 3), dtype=np.uint8) for _ in cast(list[int], indices)]

    dataset = TemporalSnippetDataset(
        [record(2), record(4)],
        {1: tmp_path / "video.mp4"},
        sampling=SamplingConfig(3, (1,), (1.0,)),
        image_size=6,
        training=False,
        fixed_stride=1,
        frame_loader=frames,
    )
    item = dataset[1]
    assert item["frame_indices"].tolist() == [2, 3, 4]
    assert item["supervised"].tolist() == [True, False, True]
    assert item["frames"].shape == (3, 3, 6, 6)
    assert item["target_masks"].shape == (3, 1, 6, 6)


def test_cpu_training_smoke() -> None:
    detector, head, flow, temporal = TinyDetector(), TinyHead(), ZeroFlow(), TinyTemporal()
    configure_trainability(detector, head, flow, temporal, fine_tune_base_head=False)
    optimizer = build_optimizer(temporal, head, lr=1e-3, base_head_lr=0, weight_decay=0)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=1
    )
    frames, _, targets, supervised = _inputs()
    batch = {
        "frames": frames,
        "target_masks": targets,
        "supervised": supervised,
        "object_boxes": torch.rand(2, 4, 4),
        "text_embeddings": torch.rand(2, 2, 4),
    }
    args = argparse.Namespace(
        base_head_lr=0.0,
        max_steps=None,
        amp=False,
        base_loss_weight=1.0,
        bce_loss_weight=1.0,
        dice_loss_weight=1.0,
        dice_smooth=1.0,
        max_grad_norm=1.0,
    )
    summary, step = train_one_epoch(
        temporal,
        detector,
        head,
        flow,
        [batch],
        optimizer,
        scheduler,
        device=torch.device("cpu"),
        scaler=None,
        args=args,
        epoch=0,
        global_step=0,
    )
    assert step == 1
    assert summary["loss"] > 0
