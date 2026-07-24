from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(value: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
