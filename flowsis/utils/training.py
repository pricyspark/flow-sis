from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler # Ugly and not what docs do, but type checker throw false positive otherwise
from torch.amp.autocast_mode import autocast # Ugly and not what docs do, but type checker throw false positive otherwise

from flowsis.rtdetrv2 import RTDetrV2


@dataclass
class ResumeState:
    checkpoint_dir: Path
    epoch: int
    global_step: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_amp_enabled(*, enabled: bool, device: torch.device) -> bool:
    return enabled and device.type == "cuda"


def build_autocast_context(
    *, enabled: bool, device: torch.device
) -> autocast | nullcontext[None]:
    if is_amp_enabled(enabled=enabled, device=device):
        return autocast("cuda")
    return nullcontext()


def build_grad_scaler(
    *, enabled: bool, device: torch.device
) -> GradScaler | None:
    if not is_amp_enabled(enabled=enabled, device=device):
        return
    return GradScaler("cuda")


def resolve_resume_checkpoint(resume_from: str | Path | None) -> Path | None:
    # TODO: check correctness
    if resume_from is None:
        return

    path = Path(resume_from)
    if not path.exists():
        raise FileNotFoundError(f"Resume path does not exist: {path}")
    if path.is_file():
        raise ValueError("Expected --resume_from to point to a checkpoint directory or output directory.")

    last_checkpoint = path / "last_checkpoint"
    if last_checkpoint.exists():
        checkpoint_path = Path(last_checkpoint.read_text().strip())
        if not checkpoint_path.is_absolute():
            checkpoint_path = path / checkpoint_path
        if checkpoint_path.exists():
            return checkpoint_path

    if (path / "training_state.pt").exists():
        return path

    checkpoints = sorted(candidate for candidate in path.glob("checkpoint-*") if candidate.is_dir())
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(f"Could not find a checkpoint inside {path}")


def save_checkpoint(
    model: RTDetrV2,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    *,
    epoch: int,
    global_step: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Path:
    # TODO: check correctness
    checkpoint_dir = output_dir / f"checkpoint-{global_step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        checkpoint_dir / "training_state.pt",
    )
    (output_dir / "last_checkpoint").write_text(str(checkpoint_dir.resolve()))
    return checkpoint_dir


def load_training_state(
    checkpoint_dir: Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> ResumeState:
    # TODO: check correctness
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing training state: {state_path}")

    state: dict[str, Any] = torch.load(state_path, map_location=map_location)
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
