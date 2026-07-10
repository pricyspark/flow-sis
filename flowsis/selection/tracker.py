from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .association import assign_detections, box_center, normalized_motion
from .types import Box, Track, TrackDetection, TrackState


@dataclass(frozen=True)
class TrackerConfig:
    min_hits: int = 2
    max_age: int = 5
    min_iou: float = 0.05
    max_motion: float = 0.75
    max_area_change: float = 0.6
    label_mismatch_cost: float = 0.75
    stationary_threshold: float = 0.04
    velocity_momentum: float = 0.7
    max_history: int = 30


def _shift_box(box: Box, velocity: tuple[float, float]) -> Box:
    vx, vy = velocity
    x1, y1, x2, y2 = box
    return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)


def detections_from_mapping(detections: Mapping[str, Any]) -> list[TrackDetection]:
    boxes = detections["boxes"].detach().cpu().tolist()
    scores = detections["scores"].detach().cpu().tolist()
    labels = detections["labels"].detach().cpu().tolist()
    return [
        TrackDetection(
            box=tuple(float(value) for value in boxes[index]),
            score=float(scores[index]),
            label=int(labels[index]),
            index=index,
        )
        for index in range(len(scores))
    ]


class SelectionTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: list[TrackState] = []
        self._next_track_id = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 0

    def update(self, detections: Sequence[TrackDetection] | Mapping[str, Any]) -> list[Track]:
        normalized_detections = self._normalize_detections(detections)
        self._predict_tracks()

        matches, unmatched_tracks, unmatched_detections = assign_detections(
            self._tracks,
            normalized_detections,
            min_iou=self.config.min_iou,
            max_motion=self.config.max_motion,
            max_area_change=self.config.max_area_change,
            label_mismatch_cost=self.config.label_mismatch_cost,
        )

        for track_index, detection_index in matches:
            self._update_track(self._tracks[track_index], normalized_detections[detection_index])

        for track_index in unmatched_tracks:
            track = self._tracks[track_index]
            track.misses += 1

        for detection_index in unmatched_detections:
            self._start_track(normalized_detections[detection_index])

        self._tracks = [track for track in self._tracks if track.misses <= self.config.max_age]
        return [self._to_public_track(track) for track in self._tracks]

    def confirmed_tracks(self) -> list[Track]:
        return [self._to_public_track(track) for track in self._tracks if track.hits >= self.config.min_hits]

    def _normalize_detections(
        self,
        detections: Sequence[TrackDetection] | Mapping[str, Any],
    ) -> list[TrackDetection]:
        if isinstance(detections, Mapping):
            return detections_from_mapping(detections)
        return list(detections)

    def _predict_tracks(self) -> None:
        for track in self._tracks:
            track.box = _shift_box(track.box, track.velocity)
            track.age += 1

    def _update_track(self, track: TrackState, detection: TrackDetection) -> None:
        previous_box = track.box
        prev_x, prev_y = box_center(previous_box)
        next_x, next_y = box_center(detection.box)
        measured_velocity = (next_x - prev_x, next_y - prev_y)
        momentum = self.config.velocity_momentum
        track.velocity = (
            momentum * track.velocity[0] + (1.0 - momentum) * measured_velocity[0],
            momentum * track.velocity[1] + (1.0 - momentum) * measured_velocity[1],
        )
        track.box = detection.box
        track.score = detection.score
        track.score_ema = detection.score if track.score_ema == 0.0 else (0.8 * track.score_ema + 0.2 * detection.score)
        track.label = detection.label
        track.hits += 1
        track.misses = 0
        track.last_detection_index = detection.index
        if normalized_motion(previous_box, detection.box) <= self.config.stationary_threshold:
            track.stationary_frames += 1
        else:
            track.stationary_frames = 0
        track.history.append(detection)
        if len(track.history) > self.config.max_history:
            track.history = track.history[-self.config.max_history :]

    def _start_track(self, detection: TrackDetection) -> None:
        self._tracks.append(
            TrackState(
                track_id=self._next_track_id,
                box=detection.box,
                score=detection.score,
                label=detection.label,
                score_ema=detection.score,
                last_detection_index=detection.index,
                history=[detection],
            )
        )
        self._next_track_id += 1

    def _to_public_track(self, track: TrackState) -> Track:
        return Track(
            track_id=track.track_id,
            box=track.box,
            score=track.score,
            label=track.label,
            age=track.age,
            hits=track.hits,
            misses=track.misses,
            stationary_frames=track.stationary_frames,
            is_confirmed=track.hits >= self.config.min_hits,
            last_detection_index=track.last_detection_index,
            score_ema=track.score_ema,
            history=tuple(track.history),
        )


__all__ = [
    "SelectionTracker",
    "TrackerConfig",
    "detections_from_mapping",
]
