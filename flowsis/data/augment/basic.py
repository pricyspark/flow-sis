from typing import Any, Callable
import math
from PIL import Image, ImageEnhance
import numpy as np
import random
from scipy.ndimage import zoom
import random

from .common import bounding_box_union
from ..masks import mask2xywh

from flowsis.utils.common import init_rng


AugmentationStep = tuple[str, Callable, dict[str, Any]]


def _crop_example_helper(
    example,
    left,
    top,
    crop_width,
    crop_height,
    output_width,
    output_height,
    zoom_factor,
    bboxes,
):
    img: Image.Image = example["image"]
    objects = example["objects"]
    right = left + crop_width
    bottom = top + crop_height
    img = img.crop((left, top, right, bottom))
    img = img.resize((output_width, output_height), resample=Image.Resampling.LANCZOS)
    
    example["image"] = img
    example["height"] = output_height
    example["width"] = output_width
    example["modified"] = True
    
    if not objects:
        return example
    
    x1 = bboxes[:, 0] - left
    y1 = bboxes[:, 1] - top
    x2 = x1 + bboxes[:, 2]
    y2 = y1 + bboxes[:, 3]
    x1 = np.clip(x1, 0.0, crop_width)
    y1 = np.clip(y1, 0.0, crop_height)
    x2 = np.clip(x2, 0.0, crop_width)
    y2 = np.clip(y2, 0.0, crop_height)
    cropped_boxes = np.stack((x1, y1, x2 - x1, y2 - y1), axis=1) * zoom_factor
    
    for obj, bbox in zip(objects, cropped_boxes):
        obj["bbox"] = bbox.tolist()
        obj["area"] = float(bbox[2] * bbox[3])
        obj["modified"] = True
        
        if "mask" in obj:
            mask = obj["mask"]
            mask_cropped = mask[top:bottom, left:right]
            mask_resized = np.asarray(
                Image.fromarray(mask_cropped.astype(np.uint8)).resize(
                    (output_width, output_height),
                    resample=Image.Resampling.NEAREST,
                )
            ).astype(bool)
            obj["mask"] = mask_resized
            mask_bbox = mask2xywh(mask_resized)
            if mask_bbox is not None:
                obj["bbox"] = mask_bbox
                obj["area"] = float(mask_bbox[2] * mask_bbox[3])
            
    return example


def center_square_augment(example: dict[str, Any], **kwargs) -> dict[str, Any]:
    """Crop image to a square in the center. If size is not specified,
    default to largest possible square."""
    img: Image.Image = example["image"]
    objects = example["objects"]
    bboxes = np.array([obj["bbox"] for obj in objects], dtype=float)
    
    width, height = img.size
    short_edge = min(width, height)
    final_size = kwargs["crop_size"] if "crop_size" in kwargs else short_edge
    zoom_factor = final_size / short_edge
    
    left = (width - short_edge) // 2
    top = (height - short_edge) // 2
    
    example = _crop_example_helper(
        example=example,
        left=left,
        top=top,
        crop_width=short_edge,
        crop_height=short_edge,
        output_width=final_size,
        output_height=final_size,
        zoom_factor=zoom_factor,
        bboxes=bboxes,
    )
    return example


def random_square_augment(example: dict[str, Any], **kwargs) -> dict[str, Any]:
    """Crop image to a square with random possition offset. If size is
    not specified, default to largest possible square."""
    rng = init_rng(kwargs.get("rng", None), kwargs.get("seed", None))
        
    img: Image.Image = example["image"]
    objects = example["objects"]
    bboxes = np.array([obj["bbox"] for obj in objects], dtype=float)
    
    width, height = img.size
    short_edge = min(width, height)
    final_size = kwargs["crop_size"] if "crop_size" in kwargs else short_edge
    long_edge = max(width, height)
    zoom_factor = final_size / short_edge
    
    max_offset = long_edge - short_edge
    offset = 0 if max_offset <= 0 else int(rng.integers(max_offset + 1))
    if width < height:
        left = 0
        top = offset
    else:
        top = 0
        left = offset
    
    example = _crop_example_helper(
        example=example,
        left=left,
        top=top,
        crop_width=short_edge,
        crop_height=short_edge,
        output_width=final_size,
        output_height=final_size,
        zoom_factor=zoom_factor,
        bboxes=bboxes,
    )
    return example
    

def roi_square_augment(example: dict[str, Any], **kwargs) -> dict[str, Any]:
    """Crop image to a square center around the detection ground truth.
    If size is not specified, default to largest possible square."""
    img: Image.Image = example["image"]
    objects = example["objects"]
    if not objects:
        return random_square_augment(example, **kwargs)
    
    bboxes = np.array([obj["bbox"] for obj in objects], dtype=float)
    
    width, height = img.size
    short_edge = min(width, height)
    final_size = kwargs["crop_size"] if "crop_size" in kwargs else short_edge
    zoom_factor = final_size / short_edge    
    
    union_x, union_y, union_w, union_h = bounding_box_union(bboxes)
    union_center_x = union_x + union_w / 2
    union_center_y = union_y + union_h / 2
    if width < height:
        left = 0
        top = round(union_center_y - short_edge / 2)
        top = max(0, min(top, height - short_edge))
    else:
        left = round(union_center_x - short_edge / 2)
        left = max(0, min(left, width - short_edge))
        top = 0
    
    example = _crop_example_helper(
        example=example,
        left=left,
        top=top,
        crop_width=short_edge,
        crop_height=short_edge,
        output_width=final_size,
        output_height=final_size,
        zoom_factor=zoom_factor,
        bboxes=bboxes,
    )
    return example


def photometric_augment(example: dict[str, Any], **kwargs) -> dict[str, Any]:
    probability = kwargs.get("probability", 1.0)

    rng = init_rng(kwargs.get("rng", None), kwargs.get("seed", None))
    if rng.random() > probability:
        return example

    image = example["image"].convert("RGB")
    enhancer_ranges = [
        (ImageEnhance.Brightness, kwargs.get("brightness", (0.8, 1.2))),
        (ImageEnhance.Contrast, kwargs.get("contrast", (0.8, 1.2))),
        (ImageEnhance.Color, kwargs.get("color", (0.8, 1.2))),
        (ImageEnhance.Sharpness, kwargs.get("sharpness", (0.8, 1.2))),
    ]

    rng.shuffle(enhancer_ranges)

    for enhancer_type, factor_range in enhancer_ranges:
        lower, upper = factor_range
        factor = rng.uniform(lower, upper)
        image = enhancer_type(image).enhance(factor)

    example["image"] = image
    return example
