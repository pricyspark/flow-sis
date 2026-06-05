import math
import random
import numpy as np
from PIL import Image
from typing import Any
from pathlib import Path
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from numpy.typing import NDArray, ArrayLike
from collections.abc import Iterable, Callable


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
    def __init__(
        self, 
        augments: Iterable[Callable], 
        augment_kwargs: Iterable[Any]
    ):
        # TODO: maybe just dict mapping function to its kwargs
        self.augments = list(augments)
        self.augment_kwargs = augment_kwargs
        
    def __call__(self, example: dict):
        for augment, kwargs in zip(self.augments, self.augment_kwargs):
            example = augment(example, **kwargs)
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
    if len(rows) == 0:
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
    

def roi_square(example: dict, **kwargs):
    image_size = kwargs["image_size"]
    
    img: Image.Image = example["image"]
    W, H = img.size
    crop_size = min(W, H)
    zoom_factor = image_size / crop_size
    
    objects = example["objects"]
    bboxes = np.array(objects["bbox"], dtype=float)
    x1 = np.min(bboxes[:, 0])
    y1 = np.min(bboxes[:, 1])
    
    x2 = np.max(bboxes[:, 0] + bboxes[:, 2])
    y2 = np.max(bboxes[:, 1] + bboxes[:, 3])
    union_bbox_center = [(x1 + x2) / 2, (y1 + y2) / 2]
    
    if W < H:
        left = 0
        right = crop_size
        max_top = H - crop_size
        top = int(np.clip(round(union_bbox_center[1] - crop_size / 2), 0, max_top))
        bottom = top + crop_size
    else:
        top = 0
        bottom = crop_size
        max_left = W - crop_size
        left = int(np.clip(round(union_bbox_center[0] - crop_size / 2), 0, max_left))
        right = left + crop_size
    crop_bounds = (left, top, right, bottom)
        
    img = img.crop(crop_bounds)
    img = img.resize((image_size, image_size), resample=Image.Resampling.LANCZOS)

    x1 = np.clip(bboxes[:, 0] - left, 0, crop_size)
    y1 = np.clip(bboxes[:, 1] - top, 0, crop_size)
    x2 = np.clip(bboxes[:, 0] + bboxes[:, 2] - left, 0, crop_size)
    y2 = np.clip(bboxes[:, 1] + bboxes[:, 3] - top, 0, crop_size)

    bboxes = np.stack((x1, y1, x2 - x1, y2 - y1), axis=1) * zoom_factor
    objects["bbox"] = bboxes.tolist()
    
    objects["area"] = (bboxes[:, 2] * bboxes[:, 3]).tolist()
    example["image"] = img
    example["width"] = image_size
    example["height"] = image_size
    
    if "mask" in objects:
        for i, mask in enumerate(objects["mask"]):
            mask = mask[top:bottom, left:right] 
            mask = zoom(mask, zoom_factor, order=0)
            objects["mask"][i] = mask
            
    return example


def rotation_augment(example: dict, **kwargs):
    # TODO: make this work with multiple objects
    if "mask" in example["objects"]:
        mask = example["objects"]["mask"][0]
    else:
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
    
    # Get final crop dimensions
    W_crop, H_crop, dx, dy = rotate_crop_bounds(W, H, math.radians(angle))
    W_crop, H_crop = round(W_crop), round(H_crop)
    
    # Get offset 
    bbox = mask2xywh(mask_rot)
    assert bbox is not None
    x, y, w, h = bbox
    bbox_center = np.array((x + w / 2, y + h / 2))
    frame_center = np.array((W_rot / 2, H_rot / 2))
    offset = bbox_center - frame_center
    
    # Project offset
    offset_proj = capped_projection(offset, (dx, dy))
    offset_proj = np.trunc(offset_proj).astype(np.int32)
        
    # Get crop bounds
    W_diff = W_rot - W_crop
    H_diff = H_rot - H_crop
    col_start = W_diff // 2 + offset_proj[0]
    row_start = H_diff // 2 + offset_proj[1]
    col_end = col_start + W_crop
    row_end = row_start + H_crop
    if "pad" in kwargs:
        row_start += kwargs["pad"]
        col_start += kwargs["pad"]
        row_end -= kwargs["pad"]
        col_end -= kwargs["pad"]
    col_slice = slice(col_start, col_end)
    row_slice = slice(row_start, row_end)
    
    img_crop = img_rot.crop((col_start, row_start, col_end, row_end))
    mask_crop = mask_rot[row_slice, col_slice]
    cropped_bbox = mask2xywh(mask_crop)
    if cropped_bbox is not None:
        example["image"] = img_crop
        example["height"] = row_end - row_start
        example["width"] = col_end - col_start
        
        x_crop, y_crop, w_crop, h_crop = cropped_bbox
        example["objects"]["area"][0] = w_crop * h_crop
        example["objects"]["bbox"][0] = cropped_bbox
        example["objects"].setdefault("mask", []).append(mask_crop)
    
    return example


def rotate_crop_bounds(w: int, h: int, theta: float) -> tuple[float, float, float, float]:
    # Calculates dimensions for the no-padding rectange with maximum area.
    # Equivalent maximization problem:
    # 
    # 0 <= x <= w / 2, 0 <= y <= h / 2
    # 
    # y <= -tan(theta) * x + h / (2 * cos(theta))
    # y <= tan(theta + pi / 2) * x + w / (2 * sin(theta))
    # 
    # f(x, y) = x * y
    # 
    # x*, y* = argmax_(x,y)(f(x,y))
    
    theta = theta % math.pi
    reflect = False
    if theta >= math.pi / 2:
        theta = math.pi - theta
        reflect = True
        
    aspect_ratio = w / h
    c = math.cos(theta)
    s = math.sin(theta)
    if theta < math.pi / 3:
        bound = math.sin(2 * theta)
        if aspect_ratio < bound:
            x = w / (2 * c)
            y = w / (2 * s)
            
            dx = (h * s - x) / 2 # h * s / 2 - w / (4 * c)
            dy = (h * c - y) / 2 # h * c / 2 - w / (4 * s)
        elif aspect_ratio <= 1 / bound:
            denom = c * c - s * s
            x = (w * c - h * s) / denom
            y = (h * c - w * s) / denom
            dx, dy = 0, 0
        else:
            x = h / (2 * s)
            y = h / (2 * c)
            
            dx = (w * c - x) / 2
            dy = (w * s - y) / 2
    else:
        t = math.tan(theta / 2)
        k = s + c * t
        if aspect_ratio < 1 / k:
            x = w
            y = w * math.tan(theta / 2)
            
            dx = (h - w * k) * s / 2
            dy = (h - w * k) * c / 2
        elif aspect_ratio <= k:
            denom = c * c - s * s
            x = (w * c - h * s) / denom
            y = (h * c - w * s) / denom
            dx, dy = 0, 0
        else:
            x = h * math.tan(theta / 2)
            y = h
            
            dx = (w - h * k) * c / 2
            dy = (w - h * k) * s / 2
        
    dy *= -1 if reflect else 1        
    return x, y, dx, dy


def capped_projection(a: ArrayLike, b: ArrayLike):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # If zero vector, return zero vector
    if np.all(b == 0):
        return np.zeros_like(b)
    
    ab = np.sum(a * b, axis=-1, keepdims=True)
    bb = np.sum(b * b, axis=-1, keepdims=True)
    coef = np.divide(
        ab,
        bb,
        out=np.zeros_like(bb),
        where=bb != 0,
    )
    
    coef = np.clip(coef, -1., 1.)
    return coef * b
