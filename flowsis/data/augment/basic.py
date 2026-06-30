from typing import Any
import math
from PIL import Image, ImageEnhance
import numpy as np
import random
from scipy.ndimage import zoom
import random

from .common import (
    get_bboxes, 
    set_bboxes, 
    random_translate_delta, 
    apply_zoom, 
    apply_translation,
    filter_object_fields,
    get_object_masks,
    set_object_masks,
    resize_mask,
    bounding_box_union,
)

from flowsis.utils.common import init_rng
from ..classes import SampleContext


def _crop_example_helper(example, left, top, width, height, zoom_factor, bboxes):
    img: Image.Image = example["image"]
    objects = example["objects"]
    right = left + width
    bottom = top + height
    img = img.crop((left, top, right, bottom))
    img = img.resize((width, height), resample=Image.Resampling.LANCZOS)
    
    bboxes[:, 0] -= left
    bboxes[:, 1] -= top
    bboxes *= zoom_factor
    
    example["image"] = img
    example["height"] = height
    example["width"] = width
    example["modified"] = True
    for obj, bbox in zip(objects, bboxes):
        obj["bbox"] = bbox.tolist()
        obj["area"] = float(bbox[2] * bbox[3])
        obj["modified"] = True
        
        if "mask" in obj:
            mask = obj["mask"]
            mask_cropped = mask[top:bottom, left:right]
            mask_resized = np.asarray(
                Image.fromarray(mask_cropped.astype(np.uint8)).resize(
                    (width, height),
                    resample=Image.Resampling.NEAREST,
                )
            ).astype(bool)
            obj["mask"] = mask_resized
            
    return example


def center_square_augment(example: dict[str, Any], **kwargs) -> dict[str, Any]:
    img: Image.Image = example["image"]
    objects = example["objects"]
    bboxes = np.array([obj["bbox"] for obj in objects], dtype=float)
    
    width, height = img.size
    short_edge = min(width, height)
    final_size = kwargs["crop_size"] if "crop_size" in kwargs else short_edge
    zoom_factor = final_size / short_edge
    
    left = (width - final_size) // 2
    top = (height - final_size) // 2
    
    example = _crop_example_helper(
        example=example,
        left=left,
        top=top,
        width=final_size,
        height=final_size,
        zoom_factor=zoom_factor,
        bboxes=bboxes,
    )
    return example


def random_square_augment(example: dict[str, Any], **kwargs) -> dict[str, Any]:
    rng = kwargs.get("rng", None)
    seed = kwargs.get("seed", None)
    
    if rng is not None and seed is not None:
        raise ValueError("Pass either 'rng' or 'seed', not both.")
    
    if rng is None:
        rng = random.Random(seed)
        
    img: Image.Image = example["image"]
    objects = example["objects"]
    bboxes = np.array([obj["bbox"] for obj in objects], dtype=float)
    
    width, height = img.size
    short_edge = min(width, height)
    final_size = kwargs["crop_size"] if "crop_size" in kwargs else short_edge
    long_edge = max(width, height)
    zoom_factor = final_size / short_edge
    
    # TODO: this doesn't work if final_size != short_edge
    offset = rng.randint(0, long_edge - final_size)
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
        width=final_size,
        height=final_size,
        zoom_factor=zoom_factor,
        bboxes=bboxes,
    )
    return example
    

def roi_square_augment(example: dict[str, Any], **kwargs) -> dict[str, Any]:
    img: Image.Image = example["image"]
    objects = example["objects"]
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
        top = round(union_center_y - final_size / 2)
        top = max(0, min(top, height - final_size))
    else:
        left = round(union_center_x - final_size / 2)
        left = max(0, min(left, width - final_size))
        top = 0
    
    example = _crop_example_helper(
        example=example,
        left=left,
        top=top,
        width=final_size,
        height=final_size,
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

# EVERY DOWNWARDS IS BAD AND REQUIRES REIMPLEMENTATION

def crop_augment(example: dict[str, Any], **kwargs) -> dict[str, Any]:
    probability = kwargs.get("probability", 1.0)
    min_size = kwargs.get("min_size", None)
    max_size = kwargs.get("max_size", None)
    height = example["height"]
    width = example["width"]
    
    if min_size is None:
        min_size = max_size        
    if max_size is None:
        max_size = min_size
    if min_size is None and max_size is None:
        raise ValueError("Minimum and maximimum crop size cannot both be None.")
    assert min_size is not None
    assert max_size is not None

    if min_size > max_size:
        raise ValueError("Minimum crop size cannot be greater than maximimum.")

    rng = init_rng(kwargs.get("rng", None), kwargs.get("seed", None))
    if rng.random() > probability:
        return example
    
    size_range = max_size - min_size
    
    image = example["image"]
    crop_ratio = rng.random() * size_range + min_size
    crop_height = round(crop_ratio * height)
    crop_width = round(crop_ratio * width)
    
    x_offset = rng.integers(0, width - crop_width)
    y_offset = rng.integers(0, height - crop_height)
    
    example = _crop_example_helper(
        example=example,
        left=x_offset,
        top=y_offset,
        zoom_
    )
    
    


def translate_augment(example: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    probability = kwargs.get("probability", 1.0)
    if random.random() > probability:
        return example

    width = int(example["width"])
    height = int(example["height"])
    dx = kwargs.get("dx")
    dy = kwargs.get("dy")
    if dx is None:
        dx = random_translate_delta(
            max_shift=kwargs.get("max_dx"),
            max_shift_frac=kwargs.get("max_dx_frac"),
            size=width,
        )
    if dy is None:
        dy = random_translate_delta(
            max_shift=kwargs.get("max_dy"),
            max_shift_frac=kwargs.get("max_dy_frac"),
            size=height,
        )

    return apply_translation(
        example,
        dx=int(dx),
        dy=int(dy),
        fillcolor=kwargs.get("fillcolor", 0),
    )


def zoom_augment(example: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    probability = kwargs.get("probability", 1.0)
    if random.random() > probability:
        return example

    scale = kwargs.get("scale")
    if scale is None:
        scale_range = kwargs.get("scale_range", (0.8, 1.2))
        scale = random.uniform(*scale_range)

    return apply_zoom(
        example,
        scale=float(scale),
        fillcolor=kwargs.get("fillcolor", 0),
    )


def zoom_crop_augment(example: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    probability = kwargs.get("probability", 1.0)
    if random.random() > probability:
        return example
    focus_kwargs = kwargs.get("focus_kwargs", {})

    width = int(example["width"])
    height = int(example["height"])
    if width != height:
        raise ValueError(
            "zoom_crop_augment expects a square image. "
            "Run it after roi_square_augment or another square-preserving crop."
        )

    scale = kwargs.get("scale")
    if scale is None:
        scale = random.uniform(*kwargs.get("scale_range", (1.0, 1.25)))
    scale = float(scale)
    if scale < 1.0:
        raise ValueError(
            f"zoom_crop_augment only supports scale >= 1 to avoid fill pixels, received {scale}."
        )
    if math.isclose(scale, 1.0):
        return example

    crop_width = max(int(round(width / scale)), 1)
    crop_height = max(int(round(height / scale)), 1)

    focus_idx = compute_focus_index(example, **focus_kwargs)
    focus_bbox = get_bboxes(example["objects"]).astype(float, copy=False)[focus_idx]
    x1 = float(focus_bbox[0])
    y1 = float(focus_bbox[1])
    x2 = float(focus_bbox[0] + focus_bbox[2])
    y2 = float(focus_bbox[1] + focus_bbox[3])
    focus_center_x = (x1 + x2) / 2
    focus_center_y = (y1 + y2) / 2

    max_left = width - crop_width
    max_top = height - crop_height
    left = int(np.clip(round(focus_center_x - crop_width / 2), 0, max_left))
    top = int(np.clip(round(focus_center_y - crop_height / 2), 0, max_top))

    max_shift_frac = kwargs.get("max_shift_frac", 0.0)
    max_shift_x_frac = kwargs.get("max_shift_x_frac", max_shift_frac)
    max_shift_y_frac = kwargs.get("max_shift_y_frac", max_shift_frac)
    shift_x = random_translate_delta(max_shift_frac=max_shift_x_frac, size=max_left)
    shift_y = random_translate_delta(max_shift_frac=max_shift_y_frac, size=max_top)
    left = int(np.clip(left + shift_x, 0, max_left))
    top = int(np.clip(top + shift_y, 0, max_top))
    right = left + crop_width
    bottom = top + crop_height

    cropped_image = example["image"].crop((left, top, right, bottom))
    example["image"] = cropped_image.resize((width, height), resample=Image.Resampling.LANCZOS)

    bboxes = get_bboxes(example["objects"]).astype(np.float32, copy=False)
    x1 = np.clip(bboxes[:, 0] - left, 0, crop_width)
    y1 = np.clip(bboxes[:, 1] - top, 0, crop_height)
    x2 = np.clip(bboxes[:, 0] + bboxes[:, 2] - left, 0, crop_width)
    y2 = np.clip(bboxes[:, 1] + bboxes[:, 3] - top, 0, crop_height)
    keep = (x2 > x1) & (y2 > y1)
    if not keep.any():
        return example

    filter_object_fields(example["objects"], keep)
    resized_bboxes = np.stack((x1[keep], y1[keep], x2[keep] - x1[keep], y2[keep] - y1[keep]), axis=1)
    resized_bboxes[:, [0, 2]] *= width / crop_width
    resized_bboxes[:, [1, 3]] *= height / crop_height
    set_bboxes(example["objects"], resized_bboxes)

    masks = get_object_masks(example)
    if masks is not None:
        cropped_masks = [mask[top:bottom, left:right] for mask, keep_item in zip(masks, keep.tolist()) if keep_item]
        resized_masks = [resize_mask(mask, width, height) for mask in cropped_masks]
        set_object_masks(example, resized_masks)

    example["width"] = width
    example["height"] = height
    return example
