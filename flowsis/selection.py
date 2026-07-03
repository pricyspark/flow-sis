from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
from typing import Any

import torch


@dataclass(frozen=True)
class SelectionResult:
    index: int
    box: tuple[float, float, float, float]
    score: float
    label: int
    selection_score: float
    track_length: int
    matched_frames: int
    score_breakdown: dict[str, float]


@dataclass(frozen=True)
class _TrackletStats:
    index: int
    box: tuple[float, float, float, float]
    score: float
    label: int
    track_length: int
    matched_frames: int
    mean_area_score: float
    mean_confidence: float
    confidence_stability: float
    label_stability: float
    box_stability: float
    motion_stability: float
    continuity: float = 0.0


def _to_boxes(detections: Mapping[str, Any]) -> list[tuple[float, float, float, float]]:
    return [tuple(float(value) for value in box) for box in detections["boxes"].detach().cpu().tolist()]


def _to_scores(detections: Mapping[str, Any]) -> list[float]:
    return [float(score) for score in detections["scores"].detach().cpu().tolist()]


def _to_labels(detections: Mapping[str, Any]) -> list[int]:
    return [int(label) for label in detections["labels"].detach().cpu().tolist()]


def _box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _box_diagonal(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return (width * width + height * height) ** 0.5


def _intersection_over_union(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)

    inter_area = _box_area((inter_x1, inter_y1, inter_x2, inter_y2))
    if inter_area <= 0.0:
        return 0.0

    union_area = _box_area(left) + _box_area(right) - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _center_distance(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x, left_y = _box_center(left)
    right_x, right_y = _box_center(right)
    dx = left_x - right_x
    dy = left_y - right_y
    return (dx * dx + dy * dy) ** 0.5


def _normalized_motion(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    scale = max(_box_diagonal(left), _box_diagonal(right), 1.0)
    return min(_center_distance(left, right) / scale, 1.0)


def _confidence_stability(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    if len(scores) == 1:
        return 1.0

    score_tensor = torch.tensor(scores, dtype=torch.float32)
    spread = float(score_tensor.std(unbiased=False).item())
    return max(0.0, 1.0 - spread)


def _mean_or_default(values: Sequence[float], default: float) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


def _match_candidate(
    reference_box: tuple[float, float, float, float],
    reference_label: int,
    boxes: Sequence[tuple[float, float, float, float]],
    labels: Sequence[int],
    *,
    min_association_score: float,
) -> int | None:
    best_index: int | None = None
    best_score = min_association_score

    for candidate_index, (candidate_box, candidate_label) in enumerate(zip(boxes, labels)):
        iou = _intersection_over_union(reference_box, candidate_box)
        motion_penalty = _normalized_motion(reference_box, candidate_box)
        label_bonus = 0.25 if candidate_label == reference_label else 0.0
        association_score = iou + label_bonus - 0.5 * motion_penalty

        if association_score > best_score:
            best_score = association_score
            best_index = candidate_index

    return best_index


def _build_tracklet_stats(
    detections_log: Sequence[Mapping[str, Any]],
    current_index: int,
    *,
    min_association_score: float = 0.1,
) -> _TrackletStats:
    current_detections = detections_log[-1]
    current_boxes = _to_boxes(current_detections)
    current_scores = _to_scores(current_detections)
    current_labels = _to_labels(current_detections)

    current_box = current_boxes[current_index]
    current_score = current_scores[current_index]
    current_label = current_labels[current_index]

    matched_boxes = [current_box]
    matched_scores = [current_score]
    matched_labels = [current_label]
    consecutive_ious: list[float] = []
    consecutive_motion: list[float] = []

    reference_box = current_box
    reference_label = current_label

    for historical_detections in reversed(detections_log[:-1]):
        historical_boxes = _to_boxes(historical_detections)
        historical_scores = _to_scores(historical_detections)
        historical_labels = _to_labels(historical_detections)

        if not historical_boxes:
            continue

        match_index = _match_candidate(
            reference_box,
            reference_label,
            historical_boxes,
            historical_labels,
            min_association_score=min_association_score,
        )
        if match_index is None:
            continue

        matched_box = historical_boxes[match_index]
        matched_boxes.append(matched_box)
        matched_scores.append(historical_scores[match_index])
        matched_labels.append(historical_labels[match_index])
        consecutive_ious.append(_intersection_over_union(reference_box, matched_box))
        consecutive_motion.append(_normalized_motion(reference_box, matched_box))

        reference_box = matched_box
        reference_label = historical_labels[match_index]

    area_scores = [min(_box_area(box) / max(_box_area(current_box), 1.0), 1.0) for box in matched_boxes]
    label_matches = [1.0 if label == current_label else 0.0 for label in matched_labels]

    return _TrackletStats(
        index=current_index,
        box=current_box,
        score=current_score,
        label=current_label,
        track_length=len(matched_boxes),
        matched_frames=max(len(matched_boxes) - 1, 0),
        mean_area_score=_mean_or_default(area_scores, 0.0),
        mean_confidence=_mean_or_default(matched_scores, 0.0),
        confidence_stability=_confidence_stability(matched_scores),
        label_stability=_mean_or_default(label_matches, 1.0),
        box_stability=_mean_or_default(consecutive_ious, 0.0),
        motion_stability=1.0 - _mean_or_default(consecutive_motion, 1.0),
    )


def _continuity_score(
    candidate_box: tuple[float, float, float, float],
    candidate_label: int,
    previous_selection: SelectionResult,
) -> float:
    label_match = 1.0 if candidate_label == previous_selection.label else 0.0
    iou = _intersection_over_union(candidate_box, previous_selection.box)
    motion = 1.0 - _normalized_motion(candidate_box, previous_selection.box)
    return 0.45 * label_match + 0.35 * iou + 0.20 * motion


def _score_first_detection(stats: _TrackletStats) -> tuple[float, dict[str, float]]:
    breakdown = {
        "large": stats.mean_area_score,
        "stable_box": stats.box_stability,
        "stable_label": stats.label_stability,
        "high_confidence": stats.mean_confidence,
        "stable_confidence": stats.confidence_stability,
        "unmoving_box": stats.motion_stability,
    }
    weights = {
        "large": 0.22,
        "stable_box": 0.18,
        "stable_label": 0.14,
        "high_confidence": 0.22,
        "stable_confidence": 0.10,
        "unmoving_box": 0.14,
    }
    total = sum(weights[key] * breakdown[key] for key in weights)
    return total, breakdown


def _score_recurrent_detection(stats: _TrackletStats) -> tuple[float, dict[str, float]]:
    breakdown = {
        "large": stats.mean_area_score,
        "stable_box": stats.box_stability,
        "stable_label": stats.label_stability,
        "high_confidence": stats.mean_confidence,
        "stable_confidence": stats.confidence_stability,
        "unmoving_box": stats.motion_stability,
        "continuity": stats.continuity,
    }
    weights = {
        "large": 0.10,
        "stable_box": 0.14,
        "stable_label": 0.12,
        "high_confidence": 0.16,
        "stable_confidence": 0.10,
        "unmoving_box": 0.12,
        "continuity": 0.26,
    }
    total = sum(weights[key] * breakdown[key] for key in weights)
    return total, breakdown


def _empty_selection_error() -> ValueError:
    return ValueError("Expected detections_log to contain at least one frame with one detection.")


def select_first_detection(detections_log: Sequence[Mapping[str, Any]]) -> SelectionResult:
    """
    Pick the best current detection when no prior selection exists.

    The selector builds a small backward tracklet for each current-frame detection
    using box overlap plus a label-consistency bonus, then scores each tracklet
    with an interpretable weighted heuristic.
    """

    if not detections_log:
        raise _empty_selection_error()

    current_detections = detections_log[-1]
    current_scores = _to_scores(current_detections)
    if not current_scores:
        raise _empty_selection_error()

    best_result: SelectionResult | None = None
    for current_index in range(len(current_scores)):
        stats = _build_tracklet_stats(detections_log, current_index)
        selection_score, breakdown = _score_first_detection(stats)
        candidate = SelectionResult(
            index=stats.index,
            box=stats.box,
            score=stats.score,
            label=stats.label,
            selection_score=selection_score,
            track_length=stats.track_length,
            matched_frames=stats.matched_frames,
            score_breakdown=breakdown,
        )
        if best_result is None or candidate.selection_score > best_result.selection_score:
            best_result = candidate

    assert best_result is not None
    return best_result


def select_recurrant_detection(
    detections_log: Sequence[Mapping[str, Any]],
    previous_selection: SelectionResult | None = None,
) -> SelectionResult:
    """
    Pick the best current detection while preferring continuity with a prior pick.

    If `previous_selection` is not provided, this falls back to `select_first_detection`.
    """

    if previous_selection is None:
        return select_first_detection(detections_log)
    if not detections_log:
        raise _empty_selection_error()

    current_detections = detections_log[-1]
    current_scores = _to_scores(current_detections)
    current_boxes = _to_boxes(current_detections)
    current_labels = _to_labels(current_detections)
    if not current_scores:
        raise _empty_selection_error()

    best_result: SelectionResult | None = None
    for current_index in range(len(current_scores)):
        stats = _build_tracklet_stats(detections_log, current_index)
        continuity = _continuity_score(current_boxes[current_index], current_labels[current_index], previous_selection)
        stats = replace(stats, continuity=continuity)
        selection_score, breakdown = _score_recurrent_detection(stats)
        candidate = SelectionResult(
            index=stats.index,
            box=stats.box,
            score=stats.score,
            label=stats.label,
            selection_score=selection_score,
            track_length=stats.track_length,
            matched_frames=stats.matched_frames,
            score_breakdown=breakdown,
        )
        if best_result is None or candidate.selection_score > best_result.selection_score:
            best_result = candidate

    assert best_result is not None
    return best_result


def select_recurrent_detection(
    detections_log: Sequence[Mapping[str, Any]],
    previous_selection: SelectionResult | None = None,
) -> SelectionResult:
    return select_recurrant_detection(detections_log, previous_selection)


__all__ = [
    "SelectionResult",
    "select_first_detection",
    "select_recurrent_detection",
    "select_recurrant_detection",
]


__all__ = [
    "SelectionResult",
    "select_first_detection",
    "select_recurrant_detection",
]
