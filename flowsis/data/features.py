from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from flowsis.artifacts import atomic_torch_save


FEATURE_BUNDLE_FILE = "feature_bundle.pt"
FEATURE_BUNDLE_VERSION = 1
FEATURE_KIND = "projected_multiscale_encoder"


@dataclass(frozen=True)
class FeatureMetadata:
    detector_architecture: str
    detector_source: str
    image_size: int
    level_shapes: tuple[tuple[int, int, int], ...]
    feature_kind: str = FEATURE_KIND
    format_version: int = FEATURE_BUNDLE_VERSION


@dataclass(frozen=True)
class FeatureBundle:
    metadata: FeatureMetadata
    feature_maps: tuple[torch.Tensor, ...]


def save_feature_bundle(
    cache_dir: str | Path,
    feature_maps: tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    detector_architecture: str,
    detector_source: str,
    image_size: int,
) -> Path:
    maps = _normalize_feature_maps(feature_maps)
    metadata = FeatureMetadata(
        detector_architecture=detector_architecture,
        detector_source=detector_source,
        image_size=image_size,
        level_shapes=tuple(tuple(feature.shape) for feature in maps),
    )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / FEATURE_BUNDLE_FILE
    atomic_torch_save(
        {
            "metadata": asdict(metadata),
            "feature_maps": list(maps),
        },
        path,
    )
    return path


def load_feature_bundle(
    cache_dir: str | Path,
    *,
    expected_levels: int | None = None,
    expected_channels: int | None = None,
    expected_image_size: int | None = None,
) -> FeatureBundle:
    path = Path(cache_dir) / FEATURE_BUNDLE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing detector feature bundle: {path}")
    raw: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a feature bundle mapping at {path}.")
    metadata_raw = raw.get("metadata")
    if not isinstance(metadata_raw, dict):
        raise TypeError(f"Feature bundle {path} is missing metadata.")
    metadata = FeatureMetadata(
        format_version=int(metadata_raw.get("format_version", -1)),
        feature_kind=str(metadata_raw.get("feature_kind", "")),
        detector_architecture=str(metadata_raw.get("detector_architecture", "")),
        detector_source=str(metadata_raw.get("detector_source", "")),
        image_size=int(metadata_raw.get("image_size", -1)),
        level_shapes=tuple(
            tuple(int(value) for value in shape)
            for shape in metadata_raw.get("level_shapes", ())
        ),
    )
    if metadata.format_version != FEATURE_BUNDLE_VERSION:
        raise ValueError(
            f"Unsupported feature bundle version at {path}: "
            f"{metadata.format_version}."
        )
    if metadata.feature_kind != FEATURE_KIND:
        raise ValueError(
            f"Feature bundle {path} contains {metadata.feature_kind!r}, "
            f"not {FEATURE_KIND!r}."
        )

    maps = _normalize_feature_maps(raw.get("feature_maps"))
    actual_shapes = tuple(tuple(feature.shape) for feature in maps)
    if actual_shapes != metadata.level_shapes:
        raise ValueError(
            f"Feature shapes in {path} do not match its metadata: "
            f"{actual_shapes} != {metadata.level_shapes}."
        )
    if expected_levels is not None and len(maps) != expected_levels:
        raise ValueError(
            f"Expected {expected_levels} feature levels in {path}, got {len(maps)}."
        )
    if expected_channels is not None and any(
        feature.shape[0] != expected_channels for feature in maps
    ):
        channels = [int(feature.shape[0]) for feature in maps]
        raise ValueError(
            f"Expected {expected_channels} channels per feature level in {path}, "
            f"got {channels}."
        )
    if (
        expected_image_size is not None
        and metadata.image_size != expected_image_size
    ):
        raise ValueError(
            f"Feature bundle {path} was generated at image size "
            f"{metadata.image_size}, expected {expected_image_size}."
        )
    return FeatureBundle(metadata=metadata, feature_maps=maps)


def _normalize_feature_maps(feature_maps: Any) -> tuple[torch.Tensor, ...]:
    if not isinstance(feature_maps, (list, tuple)) or not feature_maps:
        raise TypeError("Expected a non-empty list or tuple of feature tensors.")
    normalized = []
    for level, feature in enumerate(feature_maps):
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"Feature level {level} is not a tensor.")
        if feature.ndim == 4 and feature.shape[0] == 1:
            feature = feature.squeeze(0)
        if feature.ndim != 3:
            raise ValueError(
                f"Expected feature level {level} shaped [C,H,W], got "
                f"{tuple(feature.shape)}."
            )
        normalized.append(feature.float())
    return tuple(normalized)
