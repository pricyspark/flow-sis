import numpy as np
import torch
from torch.utils.data import Dataset
from collections.abc import Iterable, Callable
from pathlib import Path
from numpy.typing import NDArray

DEFAULT_MASKS_DIR = Path("data/masks")

def load_binary(path) -> NDArray[np.bool_]: # TODO: this should go in a general utils file
    data = np.load(path)
    packed = data["packed"]
    shape = tuple(data["shape"])
    n_bits = np.prod(shape)
    flat = np.unpackbits(packed)[:n_bits]
    arr = flat.reshape(shape).astype(bool)
    return arr


class TransformDataset(Dataset):
    def __init__(self, base_dataset, transform: Callable):
        self.base_dataset = base_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        example = self.base_dataset[idx]
        return self.transform(example)
    
    
class AugmentationPipeline:
    def __init__(self, augments: Iterable[Callable]):
        self.augments = list(augments)
        
    def __call__(self, example):
        for augment in self.augments:
            example = augment(example)
        return example
    
    def __len__(self) -> int:
        return len(self.augments)
    
    def append(self, augment: Callable) -> None:
        self.augments.append(augment)

def load_mask(example, path=None) -> NDArray[np.bool_]:
    video_id = example["video_id"]
    frame_idx = example["frame_idx"]
    if path is None:
        path = DEFAULT_MASKS_DIR
    mask_path = path / f"{video_id}" / f"{frame_idx}.npz"
    mask = load_binary(mask_path)
    return mask

def example_augment(example):
    mask = load_mask(example)