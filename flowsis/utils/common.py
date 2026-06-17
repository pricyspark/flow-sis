import json
import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
from typing import Literal


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


def load_classes(
    classes_path: Path,
) -> tuple[
    dict[str, str],
    dict[str, int],
    dict[int, str],
]:
    with classes_path.open() as file:
        classes = json.load(file)

    vid2label, label2id, raw_id2label = classes
    id2label = {int(key): value for key, value in raw_id2label.items()}
    return vid2label, label2id, id2label

def resolve_activation(activation: Literal["GELU", "RELU"]) -> nn.Module:
    if activation == "GELU":
        return nn.GELU()
    if activation == "RELU":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation function: {activation}")
