from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import label

from ..masks import load_original_mask, mask2xywh


DEFAULT_MASKS_DIR = Path("data/masks")


def get_bboxes(objects: list[dict[str, Any]]) -> NDArray[np.float32]:
    bbox_values = [record["bbox"] for record in objects if "bbox" in record]
    if not bbox_values:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(bbox_values, dtype=np.float32)


def set_bboxes(objects: list[dict[str, Any]], bboxes: NDArray[np.float32]) -> None:
    if len(objects) != len(bboxes):
        raise ValueError(f"Expected {len(objects)} boxes, received {len(bboxes)}.")

    bbox_list = bboxes.astype(np.float32, copy=False).tolist()
    areas = (bboxes[:, 2] * bboxes[:, 3]).astype(np.float32, copy=False).tolist()
    for record, bbox, area in zip(objects, bbox_list, areas):
        record["bbox"] = bbox
        record["area"] = area


def filter_object_fields(objects: list[dict[str, Any]], keep: NDArray[np.bool_]) -> None:
    objects[:] = [record for record, keep_item in zip(objects, keep.tolist()) if keep_item]


def get_object_masks(example: dict[str, Any], *, allow_mask_load: bool = False) -> list[NDArray[np.bool_]] | None:
    objects = example["objects"]
    if objects and all("mask" in record for record in objects):
        return [np.asarray(record["mask"], dtype=bool).copy() for record in objects]
    if allow_mask_load and len(objects) == 1:
        try:
            return [np.asarray(load_original_mask(example), dtype=bool)]
        except FileNotFoundError:
            return None
    return None


def set_object_masks(example: dict[str, Any], masks: list[NDArray[np.bool_]]) -> None:
    objects = example["objects"]
    if len(objects) != len(masks):
        raise ValueError(f"Expected {len(objects)} masks, received {len(masks)}.")
    for record, mask in zip(objects, masks):
        record["mask"] = np.asarray(mask, dtype=bool).copy()


def _get_object_metric(
    objects: list[dict[str, Any]],
    key: str,
    *,
    count: int,
) -> NDArray[np.float32] | None:
    if len(objects) != count:
        return None
    if any(key not in record for record in objects):
        return None
    values = [record[key] for record in objects]
    metric = np.asarray(values, dtype=np.float32)
    if metric.ndim != 1 or len(metric) != count:
        return None
    return metric


def _normalize_scores(values: NDArray[np.float32]) -> NDArray[np.float32]:
    if len(values) == 0:
        return values

    min_value = float(values.min())
    max_value = float(values.max())
    value_range = max_value - min_value
    if math.isclose(value_range, 0):
        return np.ones(len(values), dtype=np.float32)

    return (values - min_value) / value_range


def compute_focus_index(
    example: dict[str, Any],
    *,
    confidence_key: str = "confidence",
    stability_key: str = "stability",
    visible_frac_key: str = "visible_frac",
    largest_component_frac_key: str = "largest_component_frac",
    min_area_frac: float = 0.01,
    min_visible_frac: float | None = None,
    min_largest_component_frac: float | None = None,
    min_confidence: float | None = None,
    min_stability: float | None = None,
    confidence_weight: float = 0.30,
    stability_weight: float = 0.45,
    size_weight: float = 0.15,
    center_weight: float = 0.10,
) -> int:
    bboxes = get_bboxes(example["objects"])
    if len(bboxes) == 0:
        raise ValueError("compute_focus_index requires at least one bounding box.")
    if len(bboxes) == 1:
        return 0

    width = max(float(example["width"]), 1.0)
    height = max(float(example["height"]), 1.0)
    image_area = width * height

    area_frac = np.clip((bboxes[:, 2] * bboxes[:, 3]) / image_area, 0.0, 1.0)
    valid = area_frac >= float(min_area_frac)

    objects = example["objects"]
    visible_frac = _get_object_metric(objects, visible_frac_key, count=len(bboxes))
    if visible_frac is not None:
        visible_frac = np.clip(visible_frac, 0.0, 1.0)
        if min_visible_frac is not None:
            valid &= visible_frac >= float(min_visible_frac)

    largest_component_frac = _get_object_metric(objects, largest_component_frac_key, count=len(bboxes))
    if largest_component_frac is not None:
        largest_component_frac = np.clip(largest_component_frac, 0.0, 1.0)
        if min_largest_component_frac is not None:
            valid &= largest_component_frac >= float(min_largest_component_frac)

    confidence = _get_object_metric(objects, confidence_key, count=len(bboxes))
    if confidence is not None:
        confidence = np.clip(confidence, 0.0, 1.0)
        if min_confidence is not None:
            valid &= confidence >= float(min_confidence)

    stability = _get_object_metric(objects, stability_key, count=len(bboxes))
    if stability is not None:
        stability = np.clip(stability, 0.0, 1.0)
        if min_stability is not None:
            valid &= stability >= float(min_stability)

    log_area = np.log(np.clip(area_frac, 1e-6, 1.0))
    size_score = _normalize_scores(log_area)

    center_x = bboxes[:, 0] + bboxes[:, 2] / 2
    center_y = bboxes[:, 1] + bboxes[:, 3] / 2
    dx = (center_x - width / 2) / max(width / 2, 1.0)
    dy = (center_y - height / 2) / max(height / 2, 1.0)
    center_distance = np.sqrt(dx * dx + dy * dy) / math.sqrt(2.0)
    center_score = 1.0 - np.clip(center_distance, 0.0, 1.0)

    score = np.zeros(len(bboxes), dtype=np.float32)
    total_weight = 0.0
    if stability is not None and stability_weight > 0:
        score += float(stability_weight) * stability
        total_weight += float(stability_weight)
    if confidence is not None and confidence_weight > 0:
        score += float(confidence_weight) * confidence
        total_weight += float(confidence_weight)
    if size_weight > 0:
        score += float(size_weight) * size_score
        total_weight += float(size_weight)
    if center_weight > 0:
        score += float(center_weight) * center_score
        total_weight += float(center_weight)
    if total_weight > 0:
        score /= total_weight

    candidate_indices = np.flatnonzero(valid)
    if len(candidate_indices) == 0:
        candidate_indices = np.arange(len(bboxes))

    candidate_scores = score[candidate_indices]
    best_local_index = int(np.argmax(candidate_scores))
    tied_mask = np.isclose(candidate_scores, candidate_scores[best_local_index])
    tied_indices = candidate_indices[tied_mask]
    if len(tied_indices) == 1:
        return int(tied_indices[0])

    tie_break = area_frac[tied_indices] + 1e-3 * center_score[tied_indices]
    return int(tied_indices[int(np.argmax(tie_break))])


def rectangular_masks_from_bboxes(
    objects: list[dict[str, Any]],
    *,
    width: int,
    height: int,
) -> list[NDArray[np.bool_]]:
    masks: list[NDArray[np.bool_]] = []
    for x, y, box_width, box_height in get_bboxes(objects):
        mask = np.zeros((height, width), dtype=bool)
        x1 = max(int(math.floor(x)), 0)
        y1 = max(int(math.floor(y)), 0)
        x2 = min(int(math.ceil(x + box_width)), width)
        y2 = min(int(math.ceil(y + box_height)), height)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
        masks.append(mask)
    return masks


def _clip_xyxy_boxes(
    xyxy_boxes: NDArray[np.float32],
    *,
    width: int,
    height: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    clipped = xyxy_boxes.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
    clipped[:, 2] = np.clip(clipped[:, 2], 0, width)
    clipped[:, 3] = np.clip(clipped[:, 3], 0, height)
    keep = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
    return clipped[keep], keep


def _xywh_to_xyxy(bboxes: NDArray[np.float32]) -> NDArray[np.float32]:
    if len(bboxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xyxy = bboxes.copy()
    xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
    xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]
    return xyxy


def _xyxy_to_xywh(bboxes: NDArray[np.float32]) -> NDArray[np.float32]:
    if len(bboxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xywh = bboxes.copy()
    xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
    xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
    return xywh


def _translate_mask(mask: NDArray[np.bool_], dx: int, dy: int) -> NDArray[np.bool_]:
    height, width = mask.shape
    translated = np.zeros((height, width), dtype=bool)

    src_x1 = max(0, -dx)
    src_y1 = max(0, -dy)
    src_x2 = min(width, width - dx)
    src_y2 = min(height, height - dy)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return translated

    dst_x1 = max(0, dx)
    dst_y1 = max(0, dy)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    translated[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return translated


def _centered_resize_params(
    target_width: int,
    target_height: int,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int, int, int]:
    dst_left = max((target_width - source_width) // 2, 0)
    dst_top = max((target_height - source_height) // 2, 0)
    src_left = max((source_width - target_width) // 2, 0)
    src_top = max((source_height - target_height) // 2, 0)
    paste_width = min(source_width, target_width)
    paste_height = min(source_height, target_height)
    return dst_left, dst_top, src_left, src_top, paste_width, paste_height


def _centered_compose_image(
    image: Image.Image,
    *,
    width: int,
    height: int,
    fillcolor: int | tuple[int, int, int] = 0,
) -> Image.Image:
    canvas = Image.new(image.mode, (width, height), color=fillcolor)
    dst_left, dst_top, src_left, src_top, paste_width, paste_height = _centered_resize_params(
        width,
        height,
        image.width,
        image.height,
    )
    source = image.crop((src_left, src_top, src_left + paste_width, src_top + paste_height))
    canvas.paste(source, (dst_left, dst_top))
    return canvas


def _centered_compose_mask(mask: NDArray[np.bool_], *, width: int, height: int) -> NDArray[np.bool_]:
    composed = np.zeros((height, width), dtype=bool)
    src_height, src_width = mask.shape
    dst_left, dst_top, src_left, src_top, paste_width, paste_height = _centered_resize_params(
        width,
        height,
        src_width,
        src_height,
    )
    composed[dst_top:dst_top + paste_height, dst_left:dst_left + paste_width] = mask[
        src_top:src_top + paste_height,
        src_left:src_left + paste_width,
    ]
    return composed


def resize_mask(mask: NDArray[np.bool_], width: int, height: int) -> NDArray[np.bool_]:
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = mask_image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def fit_example_to_canvas(example: dict[str, Any], *, width: int, height: int) -> dict[str, Any]:
    img: Image.Image = example["image"]
    if img.width == width and img.height == height:
        return example

    scale = min(width / img.width, height / img.height)
    resized_width = max(int(round(img.width * scale)), 1)
    resized_height = max(int(round(img.height * scale)), 1)
    example["image"] = _centered_compose_image(
        img.resize((resized_width, resized_height), resample=Image.Resampling.LANCZOS),
        width=width,
        height=height,
    )
    example["width"] = width
    example["height"] = height

    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    bboxes = get_bboxes(example["objects"])
    if len(bboxes) > 0:
        resized_bboxes = bboxes.copy()
        resized_bboxes[:, 0] = resized_bboxes[:, 0] * scale + left
        resized_bboxes[:, 1] = resized_bboxes[:, 1] * scale + top
        resized_bboxes[:, 2] *= scale
        resized_bboxes[:, 3] *= scale
        set_bboxes(example["objects"], resized_bboxes)

    masks = get_object_masks(example)
    if masks is not None:
        fitted_masks = []
        for mask in masks:
            resized_mask = resize_mask(mask, resized_width, resized_height)
            fitted_masks.append(_centered_compose_mask(resized_mask, width=width, height=height))
        set_object_masks(example, fitted_masks)

    return example


def count_connected_components(mask: NDArray[np.bool_]) -> tuple[int, float]:
    if not mask.any():
        return 0, 0.0
    labeled, num_components = label(mask)
    if num_components == 0:
        return 0, 0.0
    component_sizes = np.bincount(labeled.ravel())[1:]
    largest_component = float(component_sizes.max(initial=0))
    total_area = float(component_sizes.sum())
    largest_component_frac = 0.0 if total_area == 0 else largest_component / total_area
    return int(num_components), largest_component_frac


def sample_overlay_count(
    *,
    max_overlays: int,
    distribution: str = "fixed",
    mean_overlays: float | None = None,
    geometric_p: float | None = None,
) -> int:
    if max_overlays <= 0:
        return 0
    if distribution == "fixed":
        return max_overlays
    if distribution != "geometric":
        raise ValueError(f"Unsupported overlap count distribution: {distribution}.")

    if geometric_p is None:
        if mean_overlays is None or mean_overlays <= 0:
            geometric_p = 0.5
        else:
            geometric_p = 1.0 / (mean_overlays + 1.0)
    geometric_p = float(np.clip(geometric_p, 1e-6, 1.0))

    count = 0
    while count < max_overlays and random.random() > geometric_p:
        count += 1
    return count


def apply_translation(
    example: dict[str, Any],
    *,
    dx: int,
    dy: int,
    fillcolor: int | tuple[int, int, int] = 0,
) -> dict[str, Any]:
    if dx == 0 and dy == 0:
        return example

    width = int(example["width"])
    height = int(example["height"])
    transformed = copy.deepcopy(example)

    transformed["image"] = transformed["image"].transform(
        (width, height),
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.BICUBIC,
        fillcolor=fillcolor,
    )

    masks = get_object_masks(transformed)
    if masks is not None:
        translated_masks = [_translate_mask(mask, dx, dy) for mask in masks]
        keep = np.asarray([mask.any() for mask in translated_masks], dtype=bool)
        if not keep.any():
            return example
        filter_object_fields(transformed["objects"], keep)
        translated_masks = [mask for mask, keep_item in zip(translated_masks, keep.tolist()) if keep_item]
        set_object_masks(transformed, translated_masks)
        mask_bboxes = [mask2xywh(mask) for mask in translated_masks]
        bbox_array = np.asarray(mask_bboxes, dtype=np.float32)
        set_bboxes(transformed["objects"], bbox_array)
        return transformed

    bboxes = get_bboxes(transformed["objects"])
    if len(bboxes) == 0:
        return transformed

    xyxy_boxes = _xywh_to_xyxy(bboxes)
    xyxy_boxes[:, [0, 2]] += dx
    xyxy_boxes[:, [1, 3]] += dy
    clipped_boxes, keep = _clip_xyxy_boxes(xyxy_boxes, width=width, height=height)
    if not keep.any():
        return example
    filter_object_fields(transformed["objects"], keep)
    set_bboxes(transformed["objects"], _xyxy_to_xywh(clipped_boxes))
    return transformed


def apply_zoom(
    example: dict[str, Any],
    *,
    scale: float,
    fillcolor: int | tuple[int, int, int] = 0,
) -> dict[str, Any]:
    if scale <= 0:
        raise ValueError(f"Zoom scale must be positive, received {scale}.")
    if math.isclose(scale, 1.0):
        return example

    width = int(example["width"])
    height = int(example["height"])
    transformed = copy.deepcopy(example)

    scaled_width = max(int(round(width * scale)), 1)
    scaled_height = max(int(round(height * scale)), 1)
    scaled_image = transformed["image"].resize((scaled_width, scaled_height), resample=Image.Resampling.LANCZOS)
    transformed["image"] = _centered_compose_image(
        scaled_image,
        width=width,
        height=height,
        fillcolor=fillcolor,
    )

    dst_left, dst_top, src_left, src_top, _, _ = _centered_resize_params(
        width,
        height,
        scaled_width,
        scaled_height,
    )
    scale_x = scaled_width / width
    scale_y = scaled_height / height

    masks = get_object_masks(transformed)
    if masks is not None:
        zoomed_masks = []
        for mask in masks:
            resized_mask = resize_mask(mask, scaled_width, scaled_height)
            zoomed_masks.append(_centered_compose_mask(resized_mask, width=width, height=height))
        keep = np.asarray([mask.any() for mask in zoomed_masks], dtype=bool)
        if not keep.any():
            return example
        filter_object_fields(transformed["objects"], keep)
        zoomed_masks = [mask for mask, keep_item in zip(zoomed_masks, keep.tolist()) if keep_item]
        set_object_masks(transformed, zoomed_masks)
        mask_bboxes = [mask2xywh(mask) for mask in zoomed_masks]
        bbox_array = np.asarray(mask_bboxes, dtype=np.float32)
        set_bboxes(transformed["objects"], bbox_array)
        return transformed

    bboxes = get_bboxes(transformed["objects"])
    if len(bboxes) == 0:
        return transformed

    xyxy_boxes = _xywh_to_xyxy(bboxes)
    xyxy_boxes[:, [0, 2]] = xyxy_boxes[:, [0, 2]] * scale_x + dst_left - src_left
    xyxy_boxes[:, [1, 3]] = xyxy_boxes[:, [1, 3]] * scale_y + dst_top - src_top
    clipped_boxes, keep = _clip_xyxy_boxes(xyxy_boxes, width=width, height=height)
    if not keep.any():
        return example
    filter_object_fields(transformed["objects"], keep)
    set_bboxes(transformed["objects"], _xyxy_to_xywh(clipped_boxes))
    return transformed


def random_translate_delta(
    *,
    max_shift: int | None = None,
    max_shift_frac: float | None = None,
    size: int,
) -> int:
    if max_shift is not None:
        return random.randint(-max_shift, max_shift)
    if max_shift_frac is not None:
        pixel_shift = int(round(size * max_shift_frac))
        return random.randint(-pixel_shift, pixel_shift)
    return 0
