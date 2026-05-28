import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections.abc import Iterable, Callable
from pathlib import Path
from numpy.typing import NDArray
from PIL import Image
import math

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
    
    def __getitem__(self, idx: int):
        example = self.base_dataset[idx]
        return self.transform(example)
    
    
class AugmentationPipeline:
    def __init__(self, augments: Iterable[Callable]):
        self.augments = list(augments)
        
    def __call__(self, example: dict):
        for i, augment in enumerate(self.augments):
            example = augment(example, i == len(self) - 1)
        return example
    
    def __len__(self) -> int:
        return len(self.augments)
    
    def append(self, augment: Callable) -> None:
        self.augments.append(augment)


def load_mask(example: dict, path=None) -> NDArray[np.bool_]:
    video_id = example["video_id"]
    frame_idx = example["frame_idx"]
    if path is None:
        path = DEFAULT_MASKS_DIR
    mask_path = path / f"{video_id}" / f"{frame_idx}.npz"
    mask = load_binary(mask_path)
    return mask


def mask2xywh(mask: NDArray) -> list[int] | None:
    rows, cols = np.nonzero(mask)
    if not rows:
        return None
    
    x_min = cols.min()
    x_max = cols.max()
    y_min = rows.min()
    y_max = rows.max()
    
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    
    return [
        int(x_min),
        int(y_min),
        int(width),
        int(height),
    ]
    

def rotation_augment(example: dict, last: bool):
    # TODO: check if mask already saved in example before loading
    mask = load_mask(example)
    angle = random.uniform(0, 360)
    img: Image.Image = example["image"]
    W, H = img.size
    img_rot = img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=0,
    )
    W_rot, H_rot = img_rot.size
    mask_img = Image.fromarray(mask.astype(np.uint8), mode='L')
    mask_img_rot = mask_img.rotate(
        angle,
        resample=Image.Resampling.NEAREST,
        expand=True,
        fillcolor=0,
    )
    mask_rot = np.array(mask_img_rot) > 0
    
    W_crop, H_crop = rotate_crop_bounds(W, H, math.radians(angle))
    W_crop = int(W_crop // 2) * 2
    H_crop = int(H_crop // 2) * 2
    
    # TODO: calculate bias to shift rectangle to preserve ROI
    
    W_diff = W_rot - W_crop
    H_diff = H_rot - W_crop
    col_start = W_diff // 2
    row_start = H_diff // 2
    col_end = col_start + W_crop
    row_end = row_start + H_crop
    col_slice = slice(col_start, col_end)
    row_slice = slice(row_start, row_end)
    
    img_crop = img_rot.crop((col_start, row_start, col_end, row_end))
    mask_crop = mask_rot[row_slice, col_slice]
    
    # TODO: modify img and mask in example

def rotate_crop_bounds(w: int, h: int, theta: float) -> tuple[float, float]:
    theta = theta % math.pi
    if theta >= math.pi / 2:
        theta = math.pi - theta
        
    aspect_ratio = w / h
    c = math.cos(theta)
    s = math.sin(theta)
    
    if theta < math.pi / 3:
        bound = math.sin(2 * theta)
        if aspect_ratio <= bound:
            x = w / (4 * c)
            y = w / (4 * s)
        elif aspect_ratio <= 1 / bound:
            denom = 2 * (c * c - s * s)
            x = (w * c - h * s) / denom
            y = (h * c - w * s) / denom
        else:
            x = h / (4 * s)
            y = h / (4 * c)
    else:
        bound = (1 + c - 2 * c * c) / s # equiv s + c * tan(theta / 2)
        if aspect_ratio <= 1 / bound:
            x = w / 2
            y = w / 2 * math.tan(theta / 2)
        elif aspect_ratio <= bound:
            denom = 2 * (c * c - s * s)
            x = (w * c - h * s) / denom
            y = (h * c - w * s) / denom
        else:
            x = h / 2 * math.tan(theta / 2)
            y = h / 2
        
    return 2 * x, 2 * y