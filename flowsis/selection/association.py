from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Box, TrackDetection, TrackState


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def box_diagonal(box: Box) -> float:
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return (width * width + height * height) ** 0.5


def intersection_over_union(left: Box, right: Box) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)

    inter_area = box_area((inter_x1, inter_y1, inter_x2, inter_y2))
    if inter_area <= 0.0:
        return 0.0

    union_area = box_area(left) + box_area(right) - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def center_distance(left: Box, right: Box) -> float:
    left_x, left_y = box_center(left)
    right_x, right_y = box_center(right)
    dx = left_x - right_x
    dy = left_y - right_y
    return (dx * dx + dy * dy) ** 0.5


def normalized_motion(left: Box, right: Box) -> float:
    scale = max(box_diagonal(left), box_diagonal(right), 1.0)
    return min(center_distance(left, right) / scale, 1.0)


def area_ratio(left: Box, right: Box) -> float:
    left_area = box_area(left)
    right_area = box_area(right)
    scale = max(left_area, right_area, 1.0)
    return min(abs(left_area - right_area) / scale, 1.0)


def predicted_box(track: TrackState) -> Box:
    vx, vy = track.velocity
    x1, y1, x2, y2 = track.box
    return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)


def gated_cost(
    track: TrackState,
    detection: TrackDetection,
    *,
    min_iou: float,
    max_motion: float,
    max_area_change: float,
    label_mismatch_cost: float,
) -> float | None:
    prediction = predicted_box(track)
    iou = intersection_over_union(prediction, detection.box)
    motion = normalized_motion(prediction, detection.box)
    area_change = area_ratio(prediction, detection.box)

    if iou < min_iou or motion > max_motion or area_change > max_area_change:
        return None

    label_penalty = label_mismatch_cost if track.label != detection.label else 0.0
    return (1.0 - iou) + 0.35 * motion + 0.15 * area_change + label_penalty


def build_cost_matrix(
    tracks: Sequence[TrackState],
    detections: Sequence[TrackDetection],
    *,
    min_iou: float,
    max_motion: float,
    max_area_change: float,
    label_mismatch_cost: float,
) -> np.ndarray:
    cost_matrix = np.full(
        (len(tracks), len(detections)), fill_value=np.inf, dtype=np.float64
    )
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            cost = gated_cost(
                track,
                detection,
                min_iou=min_iou,
                max_motion=max_motion,
                max_area_change=max_area_change,
                label_mismatch_cost=label_mismatch_cost,
            )
            if cost is not None:
                cost_matrix[track_index, detection_index] = cost
    return cost_matrix


def assign_detections(
    tracks: Sequence[TrackState],
    detections: Sequence[TrackDetection],
    *,
    min_iou: float,
    max_motion: float,
    max_area_change: float,
    label_mismatch_cost: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost_matrix = build_cost_matrix(
        tracks,
        detections,
        min_iou=min_iou,
        max_motion=max_motion,
        max_area_change=max_area_change,
        label_mismatch_cost=label_mismatch_cost,
    )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()

    for track_index, detection_index in zip(
        row_ind.tolist(), col_ind.tolist(), strict=True
    ):
        if not np.isfinite(cost_matrix[track_index, detection_index]):
            continue
        matches.append((track_index, detection_index))
        matched_tracks.add(track_index)
        matched_detections.add(detection_index)

    unmatched_tracks = [
        index for index in range(len(tracks)) if index not in matched_tracks
    ]
    unmatched_detections = [
        index for index in range(len(detections)) if index not in matched_detections
    ]
    return matches, unmatched_tracks, unmatched_detections
