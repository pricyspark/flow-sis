from __future__ import annotations

from dataclasses import dataclass, field

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class SelectionResult:
    index: int
    box: Box
    score: float
    label: int
    selection_score: float
    track_length: int
    matched_frames: int
    score_breakdown: dict[str, float]


@dataclass(frozen=True)
class TrackDetection:
    box: Box
    score: float
    label: int
    index: int | None = None


@dataclass
class TrackState:
    track_id: int
    box: Box
    score: float
    label: int
    velocity: tuple[float, float] = (0.0, 0.0)
    age: int = 1
    hits: int = 1
    misses: int = 0
    stationary_frames: int = 0
    score_ema: float = 0.0
    last_detection_index: int | None = None
    history: list[TrackDetection] = field(default_factory=list)


@dataclass(frozen=True)
class Track:
    track_id: int
    box: Box
    score: float
    label: int
    age: int
    hits: int
    misses: int
    stationary_frames: int
    is_confirmed: bool
    last_detection_index: int | None
    score_ema: float
    history: tuple[TrackDetection, ...]
