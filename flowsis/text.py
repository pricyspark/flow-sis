import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from transformers import AutoTokenizer, Siglip2TextModel, Siglip2TextConfig, Siglip2Tokenizer
from collections.abc import Sequence

from flowsis import SigLIP2
from flowsis.utils import resolve_pretrained_source


def apply_template(label: str) -> str:
    return f"This is a photo of {label}."

