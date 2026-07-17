import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("datasets", "numpy", "pillow", "torch", "transformers"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def _git_metadata() -> dict[str, Any] | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": dirty}


def write_run_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    *,
    model_config: dict[str, Any],
    resolved: dict[str, Any] | None = None,
) -> Path:
    """Save the complete training recipe and effective model configuration."""
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "training_args": vars(args),
        "resolved": resolved or {},
        "model_config": model_config,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "git": _git_metadata(),
        },
    }
    path = output_dir / "run_config.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return path

