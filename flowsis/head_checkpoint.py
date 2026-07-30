from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from flowsis.base_head import BaseFusionHead

import torch

from flowsis.artifacts import atomic_torch_save


HEAD_CHECKPOINT_FILE = "head.pt"
HEAD_CHECKPOINT_VERSION = 1
HEAD_ARCHITECTURE = "base_fusion_head"


@dataclass(frozen=True)
class HeadCheckpoint:
    config: dict[str, Any]
    state_dict: Mapping[str, torch.Tensor]


def save_head_checkpoint(
    directory: str | Path,
    *,
    config: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / HEAD_CHECKPOINT_FILE
    atomic_torch_save(
        {
            "format_version": HEAD_CHECKPOINT_VERSION,
            "architecture": HEAD_ARCHITECTURE,
            "config": dict(config),
            "state_dict": dict(state_dict),
        },
        path,
    )
    return path


def load_head_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> HeadCheckpoint:
    path = Path(path)
    bundle = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected a head checkpoint mapping at {path}.")
    if bundle.get("format_version") != HEAD_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported head checkpoint version at {path}: "
            f"{bundle.get('format_version')!r}."
        )
    if bundle.get("architecture") != HEAD_ARCHITECTURE:
        raise ValueError(
            f"Checkpoint {path} contains architecture "
            f"{bundle.get('architecture')!r}, not {HEAD_ARCHITECTURE!r}."
        )
    config = bundle.get("config")
    state_dict = bundle.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint {path} is missing config or state_dict mappings.")
    if any(not isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise TypeError(f"Checkpoint {path} contains a non-tensor model parameter.")
    return HeadCheckpoint(config=config, state_dict=state_dict)


def resolve_head_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path

    candidates = [path / HEAD_CHECKPOINT_FILE, path / "final" / HEAD_CHECKPOINT_FILE]
    last_checkpoint = path / "last_checkpoint"
    if last_checkpoint.exists():
        checkpoint = Path(last_checkpoint.read_text().strip())
        if not checkpoint.is_absolute():
            checkpoint = path / checkpoint
        candidates.append(checkpoint / HEAD_CHECKPOINT_FILE)
    checkpoints = sorted(
        (
            checkpoint
            for checkpoint in path.glob("checkpoint-*")
            if checkpoint.is_dir()
            and checkpoint.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda checkpoint: int(checkpoint.name.removeprefix("checkpoint-")),
        reverse=True,
    )
    candidates.extend(
        checkpoint / HEAD_CHECKPOINT_FILE for checkpoint in checkpoints
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {HEAD_CHECKPOINT_FILE} under {path}.")


def load_head(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[BaseFusionHead, Path]:
    checkpoint_path = resolve_head_checkpoint(path)
    checkpoint = load_head_checkpoint(checkpoint_path, map_location="cpu")

    head = BaseFusionHead(**checkpoint.config)
    head.load_state_dict(checkpoint.state_dict)
    head.to(device)

    return head, checkpoint_path
