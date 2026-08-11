from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from flowsis.artifacts import atomic_torch_save, atomic_write_text
from flowsis.cli.common import (
    add_detector_arguments,
    append_log_event,
    dataset_from_args,
    ensure_split_exists,
    log_event,
)
from flowsis.cli.train.train_base_head import load_text_embedding
from flowsis.cli.train.training_manifest import write_run_manifest
from flowsis.head_checkpoint import load_head
from flowsis.pretrained import load_detector, load_flow_estimator
from flowsis.temporal import TemporalOutput, TemporalRefinementBranch
from flowsis.utils import build_autocast_context, build_grad_scaler, get_device, set_seed

TEMPORAL_CHECKPOINT_FILE = "temporal.pt"
TEMPORAL_CHECKPOINT_VERSION = 1
TEMPORAL_ARCHITECTURE = "temporal_refinement_branch"


@dataclass(frozen=True)
class SamplingConfig:
    length: int
    strides: tuple[int, ...]
    weights: tuple[float, ...]

    def validate(self) -> None:
        if self.length < 2:
            raise ValueError("Snippet length must be at least two.")
        if not self.strides or any(stride <= 0 for stride in self.strides):
            raise ValueError("Snippet strides must contain positive integers.")
        if len(self.strides) != len(self.weights):
            raise ValueError("Snippet stride values and weights must have equal length.")
        if any(weight < 0 for weight in self.weights) or sum(self.weights) <= 0:
            raise ValueError("Snippet stride weights must be non-negative with a positive sum.")
        if len(set(self.strides)) != len(self.strides):
            raise ValueError("Snippet strides must be unique.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "strides": list(self.strides),
            "weights": list(self.weights),
        }


def anchored_snippet_indices(anchor: int, length: int, stride: int) -> tuple[int, ...]:
    """Return chronological frame indices ending at ``anchor``."""
    if length < 2:
        raise ValueError("length must be at least two.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    start = anchor - (length - 1) * stride
    if start < 0:
        raise ValueError(
            f"Anchor {anchor} has insufficient history for length={length}, stride={stride}."
        )
    return tuple(range(start, anchor + 1, stride))


# Kept as a readable alias for callers and tests.
build_snippet_indices = anchored_snippet_indices


def sample_stride(
    strides: Sequence[int],
    weights: Sequence[float],
    *,
    generator: torch.Generator | None = None,
) -> int:
    if len(strides) != len(weights) or not strides:
        raise ValueError("strides and weights must be non-empty and have equal length.")
    probabilities = torch.as_tensor(weights, dtype=torch.float64)
    if bool((probabilities < 0).any()) or float(probabilities.sum()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum.")
    index = int(torch.multinomial(probabilities, 1, generator=generator).item())
    return int(strides[index])


def _video_ids(split: Sequence[Mapping[str, Any]]) -> set[int]:
    ids: set[int] = set()
    for example in split:
        for obj in example.get("objects", []):
            ids.add(int(obj["video_id"]))
    return ids


def validate_video_disjoint_splits(
    train_split: Sequence[Mapping[str, Any]],
    validation_split: Sequence[Mapping[str, Any]],
) -> None:
    leaked = sorted(_video_ids(train_split) & _video_ids(validation_split))
    if leaked:
        preview = ", ".join(str(video_id) for video_id in leaked[:10])
        raise ValueError(
            "Train and validation splits must be video-disjoint; shared video IDs: "
            f"{preview}."
        )


# Alternate concise name used by downstream code.
validate_video_disjoint = validate_video_disjoint_splits


def load_frame_manifest(path: str | Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    with Path(path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            video_id = int(row["video_id"])
            video_path = Path(row["video_path"]).expanduser()
            previous = paths.setdefault(video_id, video_path)
            if previous != video_path:
                raise ValueError(
                    f"Frame manifest maps video {video_id} to multiple raw paths."
                )
    if not paths:
        raise ValueError(f"Frame manifest contains no video rows: {path}")
    return paths


def _xywh_to_xyxy(box: Sequence[float]) -> torch.Tensor:
    x, y, width, height = (float(value) for value in box)
    return torch.tensor((x, y, x + width, y + height), dtype=torch.float32)


def interpolate_box(
    frame_idx: int,
    annotated_boxes: Mapping[int, torch.Tensor],
) -> torch.Tensor:
    """Interpolate a box in time, falling back to the nearest annotation."""
    if not annotated_boxes:
        raise ValueError("At least one annotated box is required.")
    if frame_idx in annotated_boxes:
        return annotated_boxes[frame_idx].clone()
    indices = sorted(annotated_boxes)
    earlier = [index for index in indices if index < frame_idx]
    later = [index for index in indices if index > frame_idx]
    if earlier and later:
        left, right = earlier[-1], later[0]
        fraction = (frame_idx - left) / (right - left)
        return torch.lerp(annotated_boxes[left], annotated_boxes[right], fraction)
    nearest = min(indices, key=lambda index: abs(index - frame_idx))
    return annotated_boxes[nearest].clone()


def _read_video_frames(path: Path, indices: Sequence[int]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open raw video: {path}")
    frames: list[np.ndarray] = []
    try:
        requested = set(indices)
        capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        for index in range(indices[0], indices[-1] + 1):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index} from {path}.")
            if index in requested:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def _load_mask(path: str | Path) -> np.ndarray:
    with np.load(path) as archive:
        key = "mask" if "mask" in archive else archive.files[0]
        return np.asarray(archive[key], dtype=np.float32)


def _transform_snippet(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray | None],
    boxes: Sequence[torch.Tensor],
    *,
    image_size: int,
    horizontal_flip: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one center-square geometry (and optional flip) to a whole clip."""
    height, width = images[0].shape[:2]
    if any(image.shape[:2] != (height, width) for image in images):
        raise ValueError("All frames in a snippet must have the same dimensions.")
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2

    frame_tensors = []
    mask_tensors = []
    transformed_boxes = []
    for image, mask, raw_box in zip(images, masks, boxes, strict=True):
        cropped = image[top : top + side, left : left + side]
        frame = torch.from_numpy(np.ascontiguousarray(cropped)).permute(2, 0, 1).float()
        frame = F.interpolate(
            frame.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False
        )[0].div_(255.0)
        if mask is None:
            mask_tensor = torch.zeros((1, image_size, image_size), dtype=torch.float32)
        else:
            cropped_mask = mask[top : top + side, left : left + side]
            mask_tensor = F.interpolate(
                torch.from_numpy(np.ascontiguousarray(cropped_mask)).float()[None, None],
                size=(image_size, image_size),
                mode="nearest",
            )[0]

        box = raw_box.clone()
        box[[0, 2]] = (box[[0, 2]] - left) / side
        box[[1, 3]] = (box[[1, 3]] - top) / side
        box.clamp_(0.0, 1.0)
        if horizontal_flip:
            frame = frame.flip(-1)
            mask_tensor = mask_tensor.flip(-1)
            old_left, old_right = box[0].clone(), box[2].clone()
            box[0], box[2] = 1.0 - old_right, 1.0 - old_left
        frame_tensors.append(frame)
        mask_tensors.append(mask_tensor)
        transformed_boxes.append(box)
    return (
        torch.stack(frame_tensors),
        torch.stack(mask_tensors),
        torch.stack(transformed_boxes),
    )


class TemporalSnippetDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """A minimal frame-level annotation view that decodes anchored video clips."""

    def __init__(
        self,
        annotations: Sequence[Mapping[str, Any]],
        video_paths: Mapping[int, Path],
        *,
        sampling: SamplingConfig,
        image_size: int,
        training: bool,
        fixed_stride: int | None = None,
        frame_loader: Callable[[Path, Sequence[int]], list[np.ndarray]] = _read_video_frames,
    ) -> None:
        sampling.validate()
        if fixed_stride is not None and fixed_stride not in sampling.strides:
            raise ValueError("fixed_stride must be one of the configured strides.")
        self.sampling = sampling
        self.image_size = image_size
        self.training = training
        self.fixed_stride = fixed_stride
        self.video_paths = dict(video_paths)
        self.frame_loader = frame_loader
        self.records: list[Mapping[str, Any]] = []
        self.by_video_category: dict[tuple[int, int], dict[int, Mapping[str, Any]]] = {}

        for example in annotations:
            for obj in example.get("objects", []):
                record = {**example, "object": obj}
                video_id = int(obj["video_id"])
                category = int(obj.get("category", 0))
                frame_idx = int(obj["frame_idx"])
                self.by_video_category.setdefault((video_id, category), {})[frame_idx] = record
                required_stride = fixed_stride or max(sampling.strides)
                if frame_idx >= (sampling.length - 1) * required_stride:
                    self.records.append(record)

        missing = sorted(
            {int(record["object"]["video_id"]) for record in self.records} - self.video_paths.keys()
        )
        if missing:
            raise KeyError(f"Frame manifest has no raw video path for video IDs: {missing[:10]}")
        if not self.records:
            raise ValueError("No annotations have enough preceding video frames for a snippet.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor_record = self.records[index]
        anchor_obj = cast(Mapping[str, Any], anchor_record["object"])
        video_id = int(anchor_obj["video_id"])
        category = int(anchor_obj.get("category", 0))
        anchor = int(anchor_obj["frame_idx"])
        stride = self.fixed_stride or sample_stride(self.sampling.strides, self.sampling.weights)
        indices = anchored_snippet_indices(anchor, self.sampling.length, stride)
        records = self.by_video_category[(video_id, category)]
        annotated_boxes = {
            timestamp: _xywh_to_xyxy(cast(Mapping[str, Any], record["object"])["bbox"])
            for timestamp, record in records.items()
        }
        images = self.frame_loader(self.video_paths[video_id], indices)
        masks = [
            None
            if timestamp not in records
            else _load_mask(cast(Mapping[str, Any], records[timestamp]["object"])["mask_path"])
            for timestamp in indices
        ]
        boxes = [interpolate_box(timestamp, annotated_boxes) for timestamp in indices]
        flip = self.training and bool(torch.rand(()) < 0.5)
        frames, target_masks, object_boxes = _transform_snippet(
            images,
            masks,
            boxes,
            image_size=self.image_size,
            horizontal_flip=flip,
        )
        return {
            "frames": frames,
            "target_masks": target_masks,
            "supervised": torch.tensor([mask is not None for mask in masks], dtype=torch.bool),
            "object_boxes": object_boxes,
            "text_embeddings": load_text_embedding(dict(anchor_obj)),
            "stride": stride,
            "video_id": video_id,
            "frame_indices": torch.tensor(indices),
        }


def collate_snippets(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ("frames", "target_masks", "supervised", "object_boxes", "text_embeddings")
    return {
        **{key: torch.stack([item[key] for item in batch]) for key in tensor_keys},
        "stride": torch.tensor([item["stride"] for item in batch], dtype=torch.long),
        "video_id": torch.tensor([item["video_id"] for item in batch], dtype=torch.long),
        "frame_indices": torch.stack([item["frame_indices"] for item in batch]),
    }


def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    dice_smooth: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return unreduced per-example binary segmentation metrics."""
    dims = tuple(range(1, logits.ndim))
    probabilities = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dims)
    intersection = (probabilities * targets).sum(dims)
    denominator = probabilities.sum(dims) + targets.sum(dims)
    dice_loss = 1.0 - (2.0 * intersection + dice_smooth) / (denominator + dice_smooth)
    prediction = probabilities >= 0.5
    target = targets >= 0.5
    hard_intersection = (prediction & target).sum(dims).float()
    prediction_area = prediction.sum(dims).float()
    target_area = target.sum(dims).float()
    union = (prediction | target).sum(dims).float()
    return {
        "bce": bce,
        "dice_loss": dice_loss,
        "dice": (2 * hard_intersection + 1) / (prediction_area + target_area + 1),
        "iou": (hard_intersection + 1) / (union + 1),
        "brier": F.mse_loss(probabilities.float(), targets.float(), reduction="none").mean(dims),
    }


def recurrent_forward(
    temporal: nn.Module,
    flow_estimator: nn.Module,
    frames: torch.Tensor,
    base_logits: torch.Tensor,
) -> list[TemporalOutput]:
    """Unroll one snippet without detaching or substituting ground truth."""
    previous_frame = frames[:, 0]
    previous_logits = base_logits[:, 0]
    outputs: list[TemporalOutput] = []
    for timestamp in range(1, frames.shape[1]):
        current_frame = frames[:, timestamp]
        with torch.no_grad():
            backward_flow = flow_estimator(current_frame, previous_frame)
        output = temporal(
            current_frame,
            previous_frame,
            base_logits[:, timestamp],
            previous_logits,
            backward_flow,
        )
        outputs.append(output)
        previous_frame = current_frame
        previous_logits = output.final_logits
    return outputs


def compute_recurrent_objective(
    temporal: nn.Module,
    flow_estimator: nn.Module,
    frames: torch.Tensor,
    base_logits: torch.Tensor,
    target_masks: torch.Tensor,
    supervised: torch.Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    dice_smooth: float = 1.0,
    base_loss_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    outputs = recurrent_forward(temporal, flow_estimator, frames, base_logits)
    batch_size = frames.shape[0]
    temporal_sums = frames.new_zeros(batch_size)
    temporal_counts = frames.new_zeros(batch_size)
    metric_values: dict[str, list[torch.Tensor]] = {
        f"{prefix}_{name}": []
        for prefix in ("base", "temporal")
        for name in ("bce", "dice_loss", "dice", "iou", "brier")
    }
    diagnostics: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "propagation_gate",
            "residual_gate",
            "absolute_residual",
            "warp_validity",
            "photometric_residual",
        )
    }

    for timestamp, output in enumerate(outputs, start=1):
        mask = supervised[:, timestamp]
        temporal_stats = segmentation_metrics(
            output.final_logits, target_masks[:, timestamp], dice_smooth=dice_smooth
        )
        per_example = bce_weight * temporal_stats["bce"] + dice_weight * temporal_stats["dice_loss"]
        temporal_sums = temporal_sums + per_example * mask
        temporal_counts = temporal_counts + mask
        if mask.any():
            base_stats = segmentation_metrics(
                base_logits[:, timestamp], target_masks[:, timestamp], dice_smooth=dice_smooth
            )
            for name in base_stats:
                metric_values[f"base_{name}"].append(base_stats[name][mask].detach())
                metric_values[f"temporal_{name}"].append(temporal_stats[name][mask].detach())
        diagnostics["propagation_gate"].append(output.propagation_gate.mean().detach())
        diagnostics["residual_gate"].append(output.residual_gate.mean().detach())
        diagnostics["absolute_residual"].append(output.logit_residual.abs().mean().detach())
        diagnostics["warp_validity"].append(output.warp_validity.mean().detach())
        diagnostics["photometric_residual"].append(output.photometric_residual.mean().detach())

    if bool((temporal_counts == 0).any()):
        raise ValueError("Every snippet must contain a supervised timestamp after initialization.")
    temporal_loss = (temporal_sums / temporal_counts).mean()
    loss = temporal_loss
    base_loss = frames.new_zeros(())
    if base_loss_weight > 0:
        base_sums = frames.new_zeros(batch_size)
        base_counts = frames.new_zeros(batch_size)
        for timestamp in range(frames.shape[1]):
            mask = supervised[:, timestamp]
            stats = segmentation_metrics(
                base_logits[:, timestamp], target_masks[:, timestamp], dice_smooth=dice_smooth
            )
            item_loss = bce_weight * stats["bce"] + dice_weight * stats["dice_loss"]
            base_sums = base_sums + item_loss * mask
            base_counts = base_counts + mask
        base_loss = (base_sums / base_counts.clamp_min(1)).mean()
        loss = loss + base_loss_weight * base_loss

    result = {"loss": loss, "temporal_loss": temporal_loss.detach(), "base_loss": base_loss.detach()}
    for name, values in metric_values.items():
        result[name] = torch.cat(values).mean() if values else frames.new_tensor(float("nan"))
    result["dice_improvement"] = result["temporal_dice"] - result["base_dice"]
    result["iou_improvement"] = result["temporal_iou"] - result["base_iou"]
    for name, values in diagnostics.items():
        result[name] = torch.stack(values).mean()
    return result


def compute_base_logits(
    detector: nn.Module,
    base_head: nn.Module,
    frames: torch.Tensor,
    text_embeddings: torch.Tensor,
    object_boxes: torch.Tensor,
    *,
    fine_tune_base_head: bool,
) -> torch.Tensor:
    batch_size, length, _, height, width = frames.shape
    flat_frames = frames.flatten(0, 1)
    with torch.no_grad():
        features = detector.extract_feature_maps(flat_frames, device_preprocess=True)
        # Detector adapters intentionally use inference_mode. Convert their outputs
        # back to ordinary tensors so a trainable head may save them for backward.
        features = tuple(feature.clone() for feature in features)
    repeated_text = text_embeddings[:, None].expand(-1, length, -1, -1).flatten(0, 1)
    flat_boxes = object_boxes.flatten(0, 1)
    context = torch.enable_grad() if fine_tune_base_head else torch.no_grad()
    with context:
        output = base_head(
            features,
            repeated_text,
            object_boxes=flat_boxes,
            mask_output_size=(height, width),
            return_intermediates=False,
        )
        logits = cast(torch.Tensor, output["mask_logits"])
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    return logits.unflatten(0, (batch_size, length))


def configure_trainability(
    detector: nn.Module,
    base_head: nn.Module,
    flow_estimator: nn.Module,
    temporal: nn.Module,
    *,
    fine_tune_base_head: bool,
) -> None:
    detector.requires_grad_(False).eval()
    flow_estimator.requires_grad_(False).eval()
    base_head.requires_grad_(fine_tune_base_head)
    base_head.train(fine_tune_base_head)
    temporal.requires_grad_(True).train()


def build_optimizer(
    temporal: nn.Module,
    base_head: nn.Module,
    *,
    lr: float,
    base_head_lr: float,
    weight_decay: float,
) -> AdamW:
    groups: list[dict[str, Any]] = [{"params": temporal.parameters(), "lr": lr}]
    if base_head_lr > 0:
        groups.append({"params": base_head.parameters(), "lr": base_head_lr})
    return AdamW(groups, weight_decay=weight_decay)


def _temporal_config(model: TemporalRefinementBranch) -> dict[str, Any]:
    channels = tuple(block.block[0].out_channels for block in model.encoder)
    return {"channels": channels, "residual_limit": model.residual_limit}


def save_temporal_checkpoint(
    directory: str | Path,
    temporal: TemporalRefinementBranch,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    global_step: int,
    scaler: torch.amp.GradScaler | None,
    detector_checkpoint: str | None,
    base_head_checkpoint: str,
    flow_checkpoint: str | None,
    image_size: int,
    sampling: SamplingConfig,
    base_head: nn.Module | None = None,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TEMPORAL_CHECKPOINT_FILE
    atomic_torch_save(
        {
            "format_version": TEMPORAL_CHECKPOINT_VERSION,
            "architecture": TEMPORAL_ARCHITECTURE,
            "temporal_config": _temporal_config(temporal),
            "temporal_state_dict": temporal.state_dict(),
            "references": {
                "detector": detector_checkpoint,
                "base_head": base_head_checkpoint,
                "flow": flow_checkpoint,
            },
            "base_head_state_dict": None if base_head is None else base_head.state_dict(),
            "image_size": image_size,
            "sampling": sampling.as_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )
    return path


def load_temporal_checkpoint(
    path: str | Path,
    temporal: TemporalRefinementBranch,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    base_head: nn.Module | None = None,
    expected_references: Mapping[str, str | None] | None = None,
    image_size: int | None = None,
    sampling: SamplingConfig | None = None,
) -> tuple[int, int]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != TEMPORAL_CHECKPOINT_VERSION:
        raise ValueError("Unsupported temporal checkpoint version.")
    if bundle.get("architecture") != TEMPORAL_ARCHITECTURE:
        raise ValueError("Checkpoint does not contain a temporal refinement branch.")
    if bundle.get("temporal_config") != _temporal_config(temporal):
        raise ValueError("Temporal architecture configuration differs from the checkpoint.")
    if expected_references is not None and bundle.get("references") != dict(expected_references):
        raise ValueError("Model checkpoint references differ from the resume checkpoint.")
    if image_size is not None and bundle.get("image_size") != image_size:
        raise ValueError("Image size differs from the resume checkpoint.")
    if sampling is not None and bundle.get("sampling") != sampling.as_dict():
        raise ValueError("Temporal sampling configuration differs from the resume checkpoint.")
    temporal.load_state_dict(bundle["temporal_state_dict"], strict=True)
    saved_base = bundle.get("base_head_state_dict")
    if base_head is not None:
        if saved_base is None:
            raise ValueError("Resume checkpoint has no fine-tuned base-head state.")
        base_head.load_state_dict(saved_base, strict=True)
    elif saved_base is not None:
        raise ValueError("Resume checkpoint fine-tuned the base head, but --base-head-lr is zero.")
    if optimizer is not None:
        optimizer.load_state_dict(bundle["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(bundle["scheduler"])
    if scaler is not None and bundle.get("scaler") is not None:
        scaler.load_state_dict(bundle["scaler"])
    return int(bundle["epoch"]), int(bundle["global_step"])


def _checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path
    marker = path / "last_checkpoint"
    if marker.exists():
        selected = Path(marker.read_text().strip())
        if not selected.is_absolute():
            selected = path / selected
        return selected / TEMPORAL_CHECKPOINT_FILE
    candidate = path / TEMPORAL_CHECKPOINT_FILE
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find {TEMPORAL_CHECKPOINT_FILE} under {path}.")


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_batch(
    temporal: nn.Module,
    detector: nn.Module,
    base_head: nn.Module,
    flow: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    use_amp: bool,
    base_head_lr: float,
    base_loss_weight: float,
    bce_weight: float,
    dice_weight: float,
    dice_smooth: float,
) -> dict[str, torch.Tensor]:
    tensors = _move_batch(batch, device)
    with build_autocast_context(enabled=use_amp, device=device):
        base_logits = compute_base_logits(
            detector,
            base_head,
            tensors["frames"],
            tensors["text_embeddings"],
            tensors["object_boxes"],
            fine_tune_base_head=base_head_lr > 0,
        )
        return compute_recurrent_objective(
            temporal,
            flow,
            tensors["frames"],
            base_logits,
            tensors["target_masks"],
            tensors["supervised"],
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            dice_smooth=dice_smooth,
            base_loss_weight=base_loss_weight if base_head_lr > 0 else 0.0,
        )


_REPORT_KEYS = (
    "loss", "temporal_loss", "base_loss", "base_bce", "base_dice_loss", "base_dice",
    "base_iou", "base_brier", "temporal_bce", "temporal_dice_loss", "temporal_dice",
    "temporal_iou", "temporal_brier", "dice_improvement", "iou_improvement",
    "propagation_gate", "residual_gate", "absolute_residual", "warp_validity",
    "photometric_residual",
)


def _averages(totals: Mapping[str, float], count: int) -> dict[str, float]:
    return {key: round(value / max(count, 1), 6) for key, value in totals.items()}


def train_one_epoch(
    temporal: TemporalRefinementBranch,
    detector: nn.Module,
    base_head: nn.Module,
    flow: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, Any], int]:
    configure_trainability(detector, base_head, flow, temporal, fine_tune_base_head=args.base_head_lr > 0)
    totals = {key: 0.0 for key in _REPORT_KEYS}
    count = 0
    started = time.perf_counter()
    for batch in loader:
        if args.max_steps is not None and global_step >= args.max_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        result = run_batch(
            temporal, detector, base_head, flow, batch, device=device, use_amp=args.amp,
            base_head_lr=args.base_head_lr, base_loss_weight=args.base_loss_weight,
            bce_weight=args.bce_loss_weight, dice_weight=args.dice_loss_weight,
            dice_smooth=args.dice_smooth,
        )
        if scaler is None:
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                args.max_grad_norm,
            )
            optimizer.step()
        else:
            scaler.scale(result["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                args.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        global_step += 1
        count += 1
        for key in totals:
            totals[key] += float(result[key].detach())
        log_event(
            "train_step",
            {
                "epoch": epoch,
                "global_step": global_step,
                "loss": round(float(result["loss"].detach()), 6),
            },
        )
    return {"epoch": epoch, "global_step": global_step, **_averages(totals, count), "time": time.perf_counter() - started}, global_step


@torch.no_grad()
def evaluate(
    temporal: TemporalRefinementBranch,
    detector: nn.Module,
    base_head: nn.Module,
    flow: nn.Module,
    loaders: Mapping[int, DataLoader],
    *,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, Any]:
    temporal.eval()
    detector.eval()
    base_head.eval()
    flow.eval()
    by_stride: dict[str, Any] = {}
    combined = {key: 0.0 for key in _REPORT_KEYS}
    combined_count = 0
    for stride, loader in loaders.items():
        totals = {key: 0.0 for key in _REPORT_KEYS}
        count = 0
        for batch in loader:
            result = run_batch(
                temporal, detector, base_head, flow, batch, device=device, use_amp=args.amp,
                base_head_lr=args.base_head_lr, base_loss_weight=args.base_loss_weight,
                bce_weight=args.bce_loss_weight, dice_weight=args.dice_loss_weight,
                dice_smooth=args.dice_smooth,
            )
            for key in totals:
                totals[key] += float(result[key])
                combined[key] += float(result[key])
            count += 1
            combined_count += 1
        by_stride[str(stride)] = _averages(totals, count)
    summary = {"epoch": epoch, **_averages(combined, combined_count), "by_stride": by_stride}
    log_event("validation_epoch", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FlowSIS temporal refinement branch.")
    parser.add_argument("--dataset-path", default="data/segmentation-dataset")
    parser.add_argument("--frame-manifest", type=Path, default=Path("data/manifests/frame_manifest.csv"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--output-dir", default="outputs/temporal")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--base-head-checkpoint", required=True)
    add_detector_arguments(parser, model_flag="--detector-model", model_dest="detector_model_source")
    parser.add_argument("--flow-model", default="raft_small")
    parser.add_argument("--flow-checkpoint", default=None)
    parser.add_argument("--snippet-length", type=int, default=6)
    parser.add_argument("--snippet-strides", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--snippet-stride-weights", type=float, nargs="+", default=(0.60, 0.35, 0.05))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-head-lr", type=float, default=0.0)
    parser.add_argument("--base-loss-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--save-logs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--temporal-channels", type=int, nargs="+", default=(32, 64, 96))
    parser.add_argument("--residual-limit", type=float, default=5.0)
    parser.add_argument("--bce-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-smooth", type=float, default=1.0)
    return parser.parse_args()


def _loader(dataset: TemporalSnippetDataset, args: argparse.Namespace, *, shuffle: bool, device: torch.device) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
        collate_fn=collate_snippets, generator=generator, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def main() -> None:
    args = parse_args()
    sampling = SamplingConfig(args.snippet_length, tuple(args.snippet_strides), tuple(args.snippet_stride_weights))
    sampling.validate()
    if args.lr <= 0 or args.base_head_lr < 0 or args.max_grad_norm <= 0:
        raise ValueError("Learning rates must be valid and max_grad_norm must be positive.")
    if args.bce_loss_weight < 0 or args.dice_loss_weight < 0 or args.dice_smooth <= 0:
        raise ValueError("Loss weights must be non-negative and dice_smooth positive.")
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    dataset = dataset_from_args(args)
    ensure_split_exists(dataset, args.train_split, role="train")
    ensure_split_exists(dataset, args.validation_split, role="validation")
    train_split = cast(Dataset, dataset[args.train_split])
    validation_split = cast(Dataset, dataset[args.validation_split])
    validate_video_disjoint_splits(train_split, validation_split)
    video_paths = load_frame_manifest(args.frame_manifest)
    train_data = TemporalSnippetDataset(
        train_split, video_paths, sampling=sampling, image_size=args.image_size, training=True
    )
    validation_loaders = {
        stride: _loader(
            TemporalSnippetDataset(
                validation_split, video_paths, sampling=sampling, image_size=args.image_size,
                training=False, fixed_stride=stride,
            ),
            args, shuffle=False, device=device,
        )
        for stride in sampling.strides
    }
    train_loader = _loader(train_data, args, shuffle=True, device=device)

    detector = load_detector(
        args.detector_model_source, architecture=args.detector_architecture,
        image_size=args.image_size, device=device,
    )
    base_head, resolved_head_path = load_head(args.base_head_checkpoint, device=device)
    flow = load_flow_estimator(args.flow_model, args.flow_checkpoint).to(device)
    temporal = TemporalRefinementBranch(
        channels=tuple(args.temporal_channels), residual_limit=args.residual_limit
    ).to(device)
    configure_trainability(detector, base_head, flow, temporal, fine_tune_base_head=args.base_head_lr > 0)
    optimizer = build_optimizer(
        temporal, base_head, lr=args.lr, base_head_lr=args.base_head_lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.max_steps or (len(train_loader) * args.epochs)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=max(total_steps, 1),
    )
    scaler = build_grad_scaler(enabled=args.amp, device=device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    references = {
        "detector": args.detector_model_source,
        "base_head": str(resolved_head_path),
        "flow": args.flow_checkpoint or args.flow_model,
    }
    run_path = write_run_manifest(
        output_dir, args, model_config=_temporal_config(temporal),
        resolved={"device": str(device), "sampling": sampling.as_dict(), "references": references},
    )
    log_event("saved_run_config", {"path": str(run_path)})

    start_epoch = global_step = 0
    if args.resume_from:
        resume_path = _checkpoint_path(Path(args.resume_from))
        start_epoch, global_step = load_temporal_checkpoint(
            resume_path, temporal, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            base_head=base_head if args.base_head_lr > 0 else None,
            expected_references=references, image_size=args.image_size, sampling=sampling,
        )
        log_event("resumed_from", {"checkpoint": str(resume_path), "epoch": start_epoch, "global_step": global_step})

    log_path = output_dir / "training_log.jsonl" if args.save_logs else None
    completed_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        train_summary, global_step = train_one_epoch(
            temporal, detector, base_head, flow, train_loader, optimizer, scheduler,
            device=device, scaler=scaler, args=args, epoch=epoch, global_step=global_step,
        )
        log_event("train_epoch", train_summary)
        validation_summary = evaluate(
            temporal, detector, base_head, flow, validation_loaders,
            device=device, args=args, epoch=epoch,
        )
        if log_path is not None:
            append_log_event(log_path, "train_epoch", train_summary)
            append_log_event(log_path, "validation_epoch", validation_summary)
        if (epoch + 1) % args.save_every_epochs == 0:
            checkpoint_dir = output_dir / f"checkpoint-{global_step:06d}"
            save_temporal_checkpoint(
                checkpoint_dir, temporal, optimizer, scheduler, epoch=epoch + 1,
                global_step=global_step, scaler=scaler,
                detector_checkpoint=references["detector"], base_head_checkpoint=references["base_head"],
                flow_checkpoint=references["flow"], image_size=args.image_size, sampling=sampling,
                base_head=base_head if args.base_head_lr > 0 else None,
            )
            atomic_write_text(output_dir / "last_checkpoint", checkpoint_dir.name + "\n")
        completed_epoch = epoch + 1
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    final_path = save_temporal_checkpoint(
        output_dir / "final", temporal, optimizer, scheduler, epoch=completed_epoch,
        global_step=global_step, scaler=scaler,
        detector_checkpoint=references["detector"], base_head_checkpoint=references["base_head"],
        flow_checkpoint=references["flow"], image_size=args.image_size, sampling=sampling,
        base_head=base_head if args.base_head_lr > 0 else None,
    )
    log_event("saved_final_model", {"path": str(final_path)})


if __name__ == "__main__":
    main()
