from typing import Any
import math
from PIL import Image, ImageEnhance
import numpy as np
import random
from scipy.ndimage import zoom


from .common import (
    get_bboxes, 
    compute_focus_index, 
    set_bboxes, 
    random_translate_delta, 
    apply_zoom, 
    apply_translation,
    filter_object_fields,
    get_object_masks,
    set_object_masks,
    resize_mask,
)

def roi_square_augment(example: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    image_size = kwargs["image_size"]
    focus_kwargs = kwargs.get("focus_kwargs", {})

    img: Image.Image = example["image"]
    width, height = img.size
    crop_size = min(width, height)
    zoom_factor = image_size / crop_size

    objects = example["objects"]
    bboxes = get_bboxes(objects).astype(float, copy=False)
    focus_idx = compute_focus_index(example, **focus_kwargs)
    focus_bbox = bboxes[focus_idx]
    x1 = float(focus_bbox[0])
    y1 = float(focus_bbox[1])
    x2 = float(focus_bbox[0] + focus_bbox[2])
    y2 = float(focus_bbox[1] + focus_bbox[3])
    union_bbox_center = [(x1 + x2) / 2, (y1 + y2) / 2]

    if width < height:
        left = 0
        right = crop_size
        max_top = height - crop_size
        top = int(np.clip(round(union_bbox_center[1] - crop_size / 2), 0, max_top))
        bottom = top + crop_size
    else:
        top = 0
        bottom = crop_size
        max_left = width - crop_size
        left = int(np.clip(round(union_bbox_center[0] - crop_size / 2), 0, max_left))
        right = left + crop_size
    crop_bounds = (left, top, right, bottom)

    img = img.crop(crop_bounds)
    img = img.resize((image_size, image_size), resample=Image.Resampling.LANCZOS)

    x1 = np.clip(bboxes[:, 0] - left, 0, crop_size)
    y1 = np.clip(bboxes[:, 1] - top, 0, crop_size)
    x2 = np.clip(bboxes[:, 0] + bboxes[:, 2] - left, 0, crop_size)
    y2 = np.clip(bboxes[:, 1] + bboxes[:, 3] - top, 0, crop_size)

    resized_bboxes = np.stack((x1, y1, x2 - x1, y2 - y1), axis=1) * zoom_factor
    set_bboxes(objects, resized_bboxes)
    example["image"] = img
    example["width"] = image_size
    example["height"] = image_size

    for object_record in objects:
        if "mask" not in object_record:
            continue
        cropped_mask = np.asarray(object_record["mask"], dtype=bool)[top:bottom, left:right]
        object_record["mask"] = zoom(cropped_mask, zoom_factor, order=0) > 0

    return example


def photometric_augment(example: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    probability = kwargs.get("probability", 1.0)
    if random.random() > probability:
        return example

    image = example["image"].convert("RGB")
    enhancer_ranges = [
        (ImageEnhance.Brightness, kwargs.get("brightness", (0.8, 1.2))),
        (ImageEnhance.Contrast, kwargs.get("contrast", (0.8, 1.2))),
        (ImageEnhance.Color, kwargs.get("color", (0.8, 1.2))),
        (ImageEnhance.Sharpness, kwargs.get("sharpness", (0.8, 1.2))),
    ]
    random.shuffle(enhancer_ranges)

    for enhancer_type, factor_range in enhancer_ranges:
        lower, upper = factor_range
        factor = random.uniform(lower, upper)
        image = enhancer_type(image).enhance(factor)

    example["image"] = image
    return example


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
