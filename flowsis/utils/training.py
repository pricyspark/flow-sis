from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.amp import GradScaler

from flowsis.artifacts import atomic_torch_save, atomic_write_text

TRAINING_STATE_VERSION = 1


class PretrainedModel(Protocol):
    def save_pretrained(self, output_dir: str | Path) -> None: ...


@dataclass
class ResumeState:
    checkpoint_dir: Path
    epoch: int
    global_step: int


def is_amp_enabled(*, enabled: bool, device: torch.device) -> bool:
    return enabled and device.type == "cuda"


def build_autocast_context(
    *, enabled: bool, device: torch.device
) -> AbstractContextManager[None]:
    if is_amp_enabled(enabled=enabled, device=device):
        return torch.autocast("cuda")
    return nullcontext()


def build_grad_scaler(*, enabled: bool, device: torch.device) -> GradScaler | None:
    if not is_amp_enabled(enabled=enabled, device=device):
        return None
    return GradScaler("cuda")


def resolve_resume_checkpoint(resume_from: str | Path | None) -> Path | None:
    if resume_from is None:
        return None

    path = Path(resume_from)
    if not path.exists():
        raise FileNotFoundError(f"Resume path does not exist: {path}")
    if path.is_file():
        raise ValueError(
            "Expected --resume_from to point to a checkpoint directory or "
            "output directory."
        )

    last_checkpoint = path / "last_checkpoint"
    if last_checkpoint.exists():
        checkpoint_path = Path(last_checkpoint.read_text().strip())
        if not checkpoint_path.is_absolute():
            checkpoint_path = path / checkpoint_path
        if checkpoint_path.exists():
            return checkpoint_path

    if (path / "training_state.pt").exists():
        return path

    checkpoints = sorted(
        (
            candidate
            for candidate in path.glob("checkpoint-*")
            if candidate.is_dir()
            and candidate.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda candidate: int(candidate.name.removeprefix("checkpoint-")),
    )
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(f"Could not find a checkpoint inside {path}")


def save_checkpoint(
    model: PretrainedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    *,
    epoch: int,
    global_step: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{global_step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    save_training_state(
        checkpoint_dir,
        optimizer,
        scheduler,
        epoch=epoch,
        global_step=global_step,
        scaler=scaler,
    )
    atomic_write_text(
        output_dir / "last_checkpoint",
        checkpoint_dir.name + "\n",
    )
    return checkpoint_dir


def save_training_state(
    checkpoint_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    global_step: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Path:
    state_path = checkpoint_dir / "training_state.pt"
    atomic_torch_save(
        {
            "format_version": TRAINING_STATE_VERSION,
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        state_path,
    )
    return state_path


def load_training_state(
    checkpoint_dir: Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> ResumeState:
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing training state: {state_path}")

    state: dict[str, Any] = torch.load(
        state_path,
        map_location=map_location,
        weights_only=True,
    )
    if state.get("format_version") != TRAINING_STATE_VERSION:
        raise ValueError(
            f"Unsupported training state version in {state_path}: "
            f"{state.get('format_version')!r}."
        )
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    return ResumeState(
        checkpoint_dir=checkpoint_dir,
        epoch=int(state["epoch"]),
        global_step=int(state["global_step"]),
    )
