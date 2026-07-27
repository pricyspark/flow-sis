from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from flowsis.pretrained import DETECTOR_ARCHITECTURES


def add_detector_arguments(
    parser: argparse.ArgumentParser,
    *,
    model_flag: str = "--model",
    model_dest: str = "model_source",
) -> None:
    parser.add_argument(
        "--detector",
        dest="detector_architecture",
        choices=DETECTOR_ARCHITECTURES,
        default=None,
        help=(
            f"Detector backend. Inferred from {model_flag} when omitted; defaults "
            "to RT-DETRv2 when both options are omitted."
        ),
    )
    parser.add_argument(
        model_flag,
        dest=model_dest,
        default=None,
        metavar="MODEL",
        help="Hugging Face Hub model ID or local model path.",
    )


def load_dataset_dict(
    dataset_path: str | Path,
    *,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
) -> DatasetDict:
    if dataset_name is not None:
        loaded = load_dataset(dataset_name, dataset_config)
    else:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        loaded = load_from_disk(str(path))
    if isinstance(loaded, Dataset):
        raise TypeError("Expected a DatasetDict, but loaded a single Dataset.")
    return loaded


def dataset_from_args(args: argparse.Namespace) -> DatasetDict:
    return load_dataset_dict(
        args.dataset_path,
        dataset_name=getattr(args, "dataset_name", None),
        dataset_config=getattr(args, "dataset_config", None),
    )


def ensure_split_exists(
    dataset: DatasetDict,
    split_name: str,
    *,
    role: str,
) -> None:
    if split_name not in dataset:
        raise KeyError(f"Missing {role} split {split_name!r}.")


def log_event(name: str, payload: dict[str, Any]) -> None:
    print(name, json.dumps(payload, sort_keys=True, default=str), flush=True)
