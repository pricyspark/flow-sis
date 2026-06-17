import copy
import math
import random
import numpy as np
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Iterable

from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import label as ndimage_label, zoom as ndimage_zoom

from .masks import load_mask, mask2xywh


DEFAULT_MASKS_DIR = Path("data/masks")


@dataclass(frozen=True)
class AugmentationContext:
    dataset: Any
    index: int

    def __len__(self) -> int:
        return len(self.dataset)

    def get_example(self, index: int) -> dict[str, Any]:
        return copy.deepcopy(self.dataset[index])

    def get_relative_example(self, offset: int, *, wrap: bool = True) -> dict[str, Any]:
        if len(self) == 0:
            raise IndexError("Cannot fetch a relative example from an empty dataset.")
        target_index = self.index + offset
        if wrap:
            target_index %= len(self)
        elif target_index < 0 or target_index >= len(self):
            raise IndexError(
                f"Relative offset {offset} from index {self.index} is out of bounds for length {len(self)}."
            )
        return self.get_example(target_index)

    def sample_examples(
        self,
        count: int,
        *,
        exclude_current: bool = True,
        replace: bool = False,
        rng: random.Random | None = None,
    ) -> list[dict[str, Any]]:
        if count <= 0:
            return []

        generator = rng if rng is not None else random
        candidate_indices = list(range(len(self)))
        if exclude_current:
            candidate_indices = [idx for idx in candidate_indices if idx != self.index]
        if not candidate_indices:
            return []

        if replace:
            selected_indices = [generator.choice(candidate_indices) for _ in range(count)]
        else:
            selected_count = min(count, len(candidate_indices))
            selected_indices = generator.sample(candidate_indices, k=selected_count)

        return [self.get_example(idx) for idx in selected_indices]


class TransformDataset(Dataset):
    def __init__(self, base_dataset, transform: Callable):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        example = copy.deepcopy(self.base_dataset[idx])
        context = AugmentationContext(self.base_dataset, idx)
        return self.transform(example, augmentation_context=context)


class AugmentationPipeline:
    def __init__(
        self,
        augments: Iterable[Callable],
        augment_kwargs: Iterable[Any],
    ):
        self.augments = list(augments)
        self.augment_kwargs = [dict(kwargs) for kwargs in augment_kwargs]

    def __call__(
        self,
        example: dict,
        augmentation_context: AugmentationContext | None = None,
    ):
        for augment, augment_kwargs in zip(self.augments, self.augment_kwargs):
            current_kwargs = dict(augment_kwargs)
            if augmentation_context is not None:
                current_kwargs.setdefault("augmentation_context", augmentation_context)
            example = augment(example, **current_kwargs)
        return example

    def __len__(self) -> int:
        return len(self.augments)

    def append(self, augment: Callable, kwargs: dict[str, Any] | None = None) -> None:
        self.augments.append(augment)
        self.augment_kwargs.append({} if kwargs is None else dict(kwargs))


def _get_bboxes(objects: dict[str, Any]) -> NDArray[np.float32]:
    bbox_values = objects.get("bbox", [])
    if not bbox_values:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(bbox_values, dtype=np.float32)


def _set_bboxes(objects: dict[str, Any], bboxes: NDArray[np.float32]) -> None:
    objects["bbox"] = bboxes.astype(np.float32, copy=False).tolist()
    objects["area"] = (bboxes[:, 2] * bboxes[:, 3]).astype(np.float32, copy=False).tolist()


def _filter_object_fields(objects: dict[str, Any], keep: NDArray[np.bool_]) -> None:
    keep_list = keep.tolist()
    for key, value in list(objects.items()):
        if isinstance(value, list) and len(value) == len(keep_list):
            objects[key] = [item for item, keep_item in zip(value, keep_list) if keep_item]


def _get_object_masks(example: dict[str, Any], *, allow_mask_load: bool = False) -> list[NDArray[np.bool_]] | None:
    objects = example["objects"]
    if "mask" in objects:
        return [np.asarray(mask, dtype=bool).copy() for mask in objects["mask"]]
    if allow_mask_load and len(objects.get("bbox", [])) == 1:
        try:
            return [np.asarray(load_mask(example), dtype=bool)]
        except FileNotFoundError:
            return None
    return None


def _set_object_masks(example: dict[str, Any], masks: list[NDArray[np.bool_]]) -> None:
    example["objects"]["mask"] = [np.asarray(mask, dtype=bool).copy() for mask in masks]


def _get_object_metric(
    objects: dict[str, Any],
    key: str,
    *,
    count: int,
) -> NDArray[np.float32] | None:
    values = objects.get(key)
    if values is None:
        return None
    metric = np.asarray(values, dtype=np.float32)
    if metric.ndim != 1 or len(metric) != count:
        return None
    return metric


def _normalize_scores(values: NDArray[np.float32]) -> NDArray[np.float32]:
    if len(values) == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
        return np.ones(len(values), dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


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
    bboxes = _get_bboxes(example["objects"])
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


def _rectangular_masks_from_bboxes(
    objects: dict[str, Any],
    *,
    width: int,
    height: int,
) -> list[NDArray[np.bool_]]:
    masks: list[NDArray[np.bool_]] = []
    for x, y, box_width, box_height in _get_bboxes(objects):
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


def _resize_mask(mask: NDArray[np.bool_], width: int, height: int) -> NDArray[np.bool_]:
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = mask_image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def _fit_example_to_canvas(example: dict[str, Any], *, width: int, height: int) -> dict[str, Any]:
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
    bboxes = _get_bboxes(example["objects"])
    if len(bboxes) > 0:
        resized_bboxes = bboxes.copy()
        resized_bboxes[:, 0] = resized_bboxes[:, 0] * scale + left
        resized_bboxes[:, 1] = resized_bboxes[:, 1] * scale + top
        resized_bboxes[:, 2] *= scale
        resized_bboxes[:, 3] *= scale
        _set_bboxes(example["objects"], resized_bboxes)

    masks = _get_object_masks(example)
    if masks is not None:
        fitted_masks = []
        for mask in masks:
            resized_mask = _resize_mask(mask, resized_width, resized_height)
            fitted_masks.append(_centered_compose_mask(resized_mask, width=width, height=height))
        _set_object_masks(example, fitted_masks)

    return example


def _count_connected_components(mask: NDArray[np.bool_]) -> tuple[int, float]:
    if not mask.any():
        return 0, 0.0
    labeled, num_components = ndimage_label(mask)
    if num_components == 0:
        return 0, 0.0
    component_sizes = np.bincount(labeled.ravel())[1:]
    largest_component = float(component_sizes.max(initial=0))
    total_area = float(component_sizes.sum())
    largest_component_frac = 0.0 if total_area == 0 else largest_component / total_area
    return int(num_components), largest_component_frac


def _sample_overlay_count(
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


def _apply_translation(
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

    masks = _get_object_masks(transformed)
    if masks is not None:
        translated_masks = [_translate_mask(mask, dx, dy) for mask in masks]
        keep = np.asarray([mask.any() for mask in translated_masks], dtype=bool)
        if not keep.any():
            return example
        _filter_object_fields(transformed["objects"], keep)
        translated_masks = [mask for mask, keep_item in zip(translated_masks, keep.tolist()) if keep_item]
        _set_object_masks(transformed, translated_masks)
        mask_bboxes = [mask2xywh(mask) for mask in translated_masks]
        bbox_array = np.asarray(mask_bboxes, dtype=np.float32)
        _set_bboxes(transformed["objects"], bbox_array)
        return transformed

    bboxes = _get_bboxes(transformed["objects"])
    if len(bboxes) == 0:
        return transformed

    xyxy_boxes = _xywh_to_xyxy(bboxes)
    xyxy_boxes[:, [0, 2]] += dx
    xyxy_boxes[:, [1, 3]] += dy
    clipped_boxes, keep = _clip_xyxy_boxes(xyxy_boxes, width=width, height=height)
    if not keep.any():
        return example
    _filter_object_fields(transformed["objects"], keep)
    _set_bboxes(transformed["objects"], _xyxy_to_xywh(clipped_boxes))
    return transformed


def _apply_zoom(
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

    masks = _get_object_masks(transformed)
    if masks is not None:
        zoomed_masks = []
        for mask in masks:
            resized_mask = _resize_mask(mask, scaled_width, scaled_height)
            zoomed_masks.append(_centered_compose_mask(resized_mask, width=width, height=height))
        keep = np.asarray([mask.any() for mask in zoomed_masks], dtype=bool)
        if not keep.any():
            return example
        _filter_object_fields(transformed["objects"], keep)
        zoomed_masks = [mask for mask, keep_item in zip(zoomed_masks, keep.tolist()) if keep_item]
        _set_object_masks(transformed, zoomed_masks)
        mask_bboxes = [mask2xywh(mask) for mask in zoomed_masks]
        bbox_array = np.asarray(mask_bboxes, dtype=np.float32)
        _set_bboxes(transformed["objects"], bbox_array)
        return transformed

    bboxes = _get_bboxes(transformed["objects"])
    if len(bboxes) == 0:
        return transformed

    xyxy_boxes = _xywh_to_xyxy(bboxes)
    xyxy_boxes[:, [0, 2]] = xyxy_boxes[:, [0, 2]] * scale_x + dst_left - src_left
    xyxy_boxes[:, [1, 3]] = xyxy_boxes[:, [1, 3]] * scale_y + dst_top - src_top
    clipped_boxes, keep = _clip_xyxy_boxes(xyxy_boxes, width=width, height=height)
    if not keep.any():
        return example
    _filter_object_fields(transformed["objects"], keep)
    _set_bboxes(transformed["objects"], _xyxy_to_xywh(clipped_boxes))
    return transformed


def _random_translate_delta(
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


def roi_square_augment(example: dict, **kwargs):
    image_size = kwargs["image_size"]
    focus_kwargs = kwargs.get("focus_kwargs", {})

    img: Image.Image = example["image"]
    width, height = img.size
    crop_size = min(width, height)
    zoom_factor = image_size / crop_size

    objects = example["objects"]
    bboxes = _get_bboxes(objects).astype(float, copy=False)
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
    _set_bboxes(objects, resized_bboxes)
    example["image"] = img
    example["width"] = image_size
    example["height"] = image_size

    if "mask" in objects:
        for i, mask in enumerate(objects["mask"]):
            cropped_mask = mask[top:bottom, left:right]
            objects["mask"][i] = ndimage_zoom(cropped_mask, zoom_factor, order=0) > 0

    return example


def rotation_augment(example: dict, **kwargs):
    focus_kwargs = kwargs.get("focus_kwargs", {})
    masks = _get_object_masks(example, allow_mask_load=True)
    if masks is None:
        masks = _rectangular_masks_from_bboxes(
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
        rotated_masks.append(np.array(mask_img_rot) > 0)

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
        _filter_object_fields(example["objects"], keep)
        cropped_masks = [mask for mask, keep_item in zip(cropped_masks, keep.tolist()) if keep_item]
        cropped_bboxes = np.asarray([mask2xywh(mask) for mask in cropped_masks], dtype=np.float32)

        example["image"] = img_crop
        example["height"] = row_end - row_start
        example["width"] = col_end - col_start
        _set_bboxes(example["objects"], cropped_bboxes)
        _set_object_masks(example, cropped_masks)

    return example


def photometric_augment(example: dict, **kwargs):
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


def translate_augment(example: dict, **kwargs):
    probability = kwargs.get("probability", 1.0)
    if random.random() > probability:
        return example

    width = int(example["width"])
    height = int(example["height"])
    dx = kwargs.get("dx")
    dy = kwargs.get("dy")
    if dx is None:
        dx = _random_translate_delta(
            max_shift=kwargs.get("max_dx"),
            max_shift_frac=kwargs.get("max_dx_frac"),
            size=width,
        )
    if dy is None:
        dy = _random_translate_delta(
            max_shift=kwargs.get("max_dy"),
            max_shift_frac=kwargs.get("max_dy_frac"),
            size=height,
        )

    return _apply_translation(
        example,
        dx=int(dx),
        dy=int(dy),
        fillcolor=kwargs.get("fillcolor", 0),
    )


def zoom_augment(example: dict, **kwargs):
    probability = kwargs.get("probability", 1.0)
    if random.random() > probability:
        return example

    scale = kwargs.get("scale")
    if scale is None:
        scale_range = kwargs.get("scale_range", (0.8, 1.2))
        scale = random.uniform(*scale_range)

    return _apply_zoom(
        example,
        scale=float(scale),
        fillcolor=kwargs.get("fillcolor", 0),
    )


def zoom_crop_augment(example: dict, **kwargs):
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
    focus_bbox = _get_bboxes(example["objects"]).astype(float, copy=False)[focus_idx]
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
    shift_x = _random_translate_delta(max_shift_frac=max_shift_x_frac, size=max_left)
    shift_y = _random_translate_delta(max_shift_frac=max_shift_y_frac, size=max_top)
    left = int(np.clip(left + shift_x, 0, max_left))
    top = int(np.clip(top + shift_y, 0, max_top))
    right = left + crop_width
    bottom = top + crop_height

    cropped_image = example["image"].crop((left, top, right, bottom))
    example["image"] = cropped_image.resize((width, height), resample=Image.Resampling.LANCZOS)

    bboxes = _get_bboxes(example["objects"]).astype(np.float32, copy=False)
    x1 = np.clip(bboxes[:, 0] - left, 0, crop_width)
    y1 = np.clip(bboxes[:, 1] - top, 0, crop_height)
    x2 = np.clip(bboxes[:, 0] + bboxes[:, 2] - left, 0, crop_width)
    y2 = np.clip(bboxes[:, 1] + bboxes[:, 3] - top, 0, crop_height)
    keep = (x2 > x1) & (y2 > y1)
    if not keep.any():
        return example

    _filter_object_fields(example["objects"], keep)
    resized_bboxes = np.stack((x1[keep], y1[keep], x2[keep] - x1[keep], y2[keep] - y1[keep]), axis=1)
    resized_bboxes[:, [0, 2]] *= width / crop_width
    resized_bboxes[:, [1, 3]] *= height / crop_height
    _set_bboxes(example["objects"], resized_bboxes)

    masks = _get_object_masks(example)
    if masks is not None:
        cropped_masks = [mask[top:bottom, left:right] for mask, keep_item in zip(masks, keep.tolist()) if keep_item]
        resized_masks = [_resize_mask(mask, width, height) for mask in cropped_masks]
        _set_object_masks(example, resized_masks)

    example["width"] = width
    example["height"] = height
    return example


def overlap_augment(example: dict, **kwargs):
    context: AugmentationContext | None = kwargs.get("augmentation_context")
    if context is None:
        raise ValueError(
            "overlap_augment requires kwargs['augmentation_context']. "
            "Use it through TransformDataset/AugmentationPipeline with an indexable dataset."
        )
    focus_kwargs = kwargs.get("focus_kwargs", {})

    probability = kwargs.get("probability", 1.0)
    if random.random() > probability:
        return example

    max_overlays = int(kwargs.get("max_overlays", kwargs.get("num_overlays", 1)))
    num_overlays = _sample_overlay_count(
        max_overlays=max_overlays,
        distribution=kwargs.get("count_distribution", "fixed"),
        mean_overlays=kwargs.get("mean_overlays"),
        geometric_p=kwargs.get("geometric_p"),
    )
    if num_overlays <= 0:
        return example

    width = int(example["width"])
    height = int(example["height"])

    base_masks = _get_object_masks(example, allow_mask_load=True)
    if base_masks is None:
        base_masks = _rectangular_masks_from_bboxes(example["objects"], width=width, height=height)
    _set_object_masks(example, base_masks)
    focus_idx = compute_focus_index(example, **focus_kwargs)
    focus_idx = max(0, min(focus_idx, len(base_masks) - 1))

    overlay_examples = context.sample_examples(
        num_overlays,
        exclude_current=kwargs.get("exclude_current", True),
        replace=kwargs.get("replace", False),
    )

    overlay_layers: list[dict[str, Any]] = []
    for overlay_example in overlay_examples:
        overlay_example = _fit_example_to_canvas(overlay_example, width=width, height=height)

        overlay_scale_range = kwargs.get("scale_range")
        if overlay_scale_range is not None:
            overlay_example = _apply_zoom(
                overlay_example,
                scale=random.uniform(*overlay_scale_range),
                fillcolor=kwargs.get("fillcolor", 0),
            )

        max_translate_frac = kwargs.get("max_translate_frac")
        max_translate_x_frac = kwargs.get("max_translate_x_frac", max_translate_frac)
        max_translate_y_frac = kwargs.get("max_translate_y_frac", max_translate_frac)
        dx = _random_translate_delta(max_shift_frac=max_translate_x_frac, size=width)
        dy = _random_translate_delta(max_shift_frac=max_translate_y_frac, size=height)
        overlay_example = _apply_translation(
            overlay_example,
            dx=dx,
            dy=dy,
                fillcolor=kwargs.get("fillcolor", 0),
            )

        overlay_masks = _get_object_masks(overlay_example, allow_mask_load=True)
        if overlay_masks is None:
            overlay_masks = _rectangular_masks_from_bboxes(
                overlay_example["objects"],
                width=width,
                height=height,
            )
        if not overlay_masks:
            continue

        overlay_layers.append(
            {
                "pixels": np.asarray(overlay_example["image"].convert("RGB")),
                "masks": [np.asarray(mask, dtype=bool) for mask in overlay_masks],
                "categories": [int(category) for category in overlay_example["objects"]["category"]],
            }
        )

    min_primary_visible_frac = float(kwargs.get("min_primary_visible_frac", 0.6))
    kept_layers = list(overlay_layers)
    while kept_layers:
        all_overlay_masks = [mask for layer in kept_layers for mask in layer["masks"]]
        if not all_overlay_masks:
            break
        occluder_union = np.any(np.stack(all_overlay_masks, axis=0), axis=0)
        focus_mask = base_masks[focus_idx]
        total_area = float(focus_mask.sum())
        visible_area = float((focus_mask & ~occluder_union).sum())
        visible_frac = 0.0 if total_area == 0 else visible_area / total_area
        if visible_frac >= min_primary_visible_frac:
            break
        kept_layers.pop()

    composed_pixels = np.asarray(example["image"].convert("RGB")).copy()
    for layer in kept_layers:
        union_mask = np.any(np.stack(layer["masks"], axis=0), axis=0)
        composed_pixels[union_mask] = layer["pixels"][union_mask]

    object_records: list[dict[str, Any]] = []
    for index, (mask, category) in enumerate(zip(base_masks, example["objects"]["category"])):
        object_records.append(
            {
                "mask": np.asarray(mask, dtype=bool),
                "category": int(category),
                "is_focus": index == focus_idx,
            }
        )
    for layer in kept_layers:
        for mask, category in zip(layer["masks"], layer["categories"]):
            object_records.append(
                {
                    "mask": np.asarray(mask, dtype=bool),
                    "category": int(category),
                    "is_focus": False,
                }
            )

    min_visible_frac = float(kwargs.get("min_visible_frac", 0.6))
    max_components = kwargs.get("max_components", 3)
    min_largest_component_frac = float(kwargs.get("min_largest_component_frac", 0.7))
    annotate_overlays = bool(kwargs.get("annotate_overlays", True))

    visible_records: list[dict[str, Any]] = []
    occluder_union = np.zeros((height, width), dtype=bool)
    for record in reversed(object_records):
        full_mask = record["mask"]
        visible_mask = full_mask & ~occluder_union
        occluder_union |= full_mask

        total_area = float(full_mask.sum())
        visible_area = float(visible_mask.sum())
        visible_frac = 0.0 if total_area == 0 else visible_area / total_area
        component_count, largest_component_frac = _count_connected_components(visible_mask)

        visible_records.append(
            {
                **record,
                "visible_mask": visible_mask,
                "visible_frac": visible_frac,
                "component_count": component_count,
                "largest_component_frac": largest_component_frac,
            }
        )
    visible_records.reverse()

    kept_masks: list[NDArray[np.bool_]] = []
    kept_categories: list[int] = []
    for record in visible_records:
        visible_mask = record["visible_mask"]
        if not visible_mask.any():
            continue
        if record["is_focus"]:
            keep_annotation = True
        else:
            keep_annotation = annotate_overlays and record["visible_frac"] >= min_visible_frac
            if max_components is not None:
                keep_annotation = keep_annotation and record["component_count"] <= int(max_components)
            keep_annotation = keep_annotation and record["largest_component_frac"] >= min_largest_component_frac
        if keep_annotation:
            kept_masks.append(visible_mask)
            kept_categories.append(record["category"])

    example["image"] = Image.fromarray(composed_pixels)
    example["width"] = width
    example["height"] = height
    example["objects"] = {
        "id": list(range(len(kept_masks))),
        "area": [],
        "bbox": [],
        "category": kept_categories,
        "mask": kept_masks,
    }
    if kept_masks:
        kept_bboxes = np.asarray([mask2xywh(mask) for mask in kept_masks], dtype=np.float32)
        _set_bboxes(example["objects"], kept_bboxes)
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


def capped_projection(a: ArrayLike, b: ArrayLike):
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
