from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from ..masks import mask2xywh
from .common import (
    filter_object_fields,
    get_object_masks,
    rectangular_masks_from_bboxes,
    set_bboxes,
    set_object_masks,
    compute_focus_index,
)


def rotation_augment(example: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    focus_kwargs = kwargs.get("focus_kwargs", {})
    masks = get_object_masks(example, allow_mask_load=True)
    if masks is None:
        masks = rectangular_masks_from_bboxes(
            example["objects"],
            width=int(example["width"]),
            height=int(example["height"]),
        )
    if not masks:
        return example

    focus_idx = compute_focus_index(example, **focus_kwargs)
    focus_idx = max(0, min(focus_idx, len(masks) - 1))

    angle = random.uniform(0, 360)
    img: Image.Image = example["image"]
    width, height = img.size
    img_rot = img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=0,
    )
    width_rot, height_rot = img_rot.size
    rotated_masks: list[NDArray[np.bool_]] = []
    for mask in masks:
        mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
        mask_img_rot = mask_img.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
            expand=True,
            fillcolor=0,
        )
        rotated_masks.append(np.asarray(mask_img_rot) > 0)

    focus_union_mask = rotated_masks[focus_idx]
    width_crop, height_crop, dx, dy = rotate_crop_bounds(width, height, math.radians(angle))
    width_crop, height_crop = round(width_crop), round(height_crop)

    bbox = mask2xywh(focus_union_mask)
    assert bbox is not None
    x, y, box_width, box_height = bbox
    bbox_center = np.array((x + box_width / 2, y + box_height / 2))
    frame_center = np.array((width_rot / 2, height_rot / 2))
    offset = bbox_center - frame_center

    offset_proj = capped_projection(offset, (dx, dy))
    offset_proj = np.trunc(offset_proj).astype(np.int32)

    width_diff = width_rot - width_crop
    height_diff = height_rot - height_crop
    col_start = width_diff // 2 + offset_proj[0]
    row_start = height_diff // 2 + offset_proj[1]
    col_end = col_start + width_crop
    row_end = row_start + height_crop
    if "pad" in kwargs:
        row_start += kwargs["pad"]
        col_start += kwargs["pad"]
        row_end -= kwargs["pad"]
        col_end -= kwargs["pad"]
    col_slice = slice(col_start, col_end)
    row_slice = slice(row_start, row_end)

    img_crop = img_rot.crop((col_start, row_start, col_end, row_end))
    cropped_masks = [mask[row_slice, col_slice] for mask in rotated_masks]
    keep = np.asarray([mask.any() for mask in cropped_masks], dtype=bool)
    if keep.any():
        filter_object_fields(example["objects"], keep)
        cropped_masks = [mask for mask, keep_item in zip(cropped_masks, keep.tolist()) if keep_item]
        cropped_bboxes = np.asarray([mask2xywh(mask) for mask in cropped_masks], dtype=np.float32)

        example["image"] = img_crop
        example["height"] = row_end - row_start
        example["width"] = col_end - col_start
        set_bboxes(example["objects"], cropped_bboxes)
        set_object_masks(example, cropped_masks)

    return example


def rotate_crop_bounds(w: int, h: int, theta: float) -> tuple[float, float, float, float]:
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

            dx = (h * s - x) / 2
            dy = (h * c - y) / 2
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


def capped_projection(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
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
    coef = np.clip(coef, -1, 1)
    return coef * b
