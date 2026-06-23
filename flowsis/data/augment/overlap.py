from __future__ import annotations

import random
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..masks import mask2xywh
from .classes import AugmentationContext
from .common import (
    apply_translation,
    apply_zoom,
    count_connected_components,
    fit_example_to_canvas,
    get_object_masks,
    random_translate_delta,
    rectangular_masks_from_bboxes,
    sample_overlay_count,
    set_bboxes,
    set_object_masks,
    compute_focus_index,
)


def overlap_augment(example: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
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
    num_overlays = sample_overlay_count(
        max_overlays=max_overlays,
        distribution=kwargs.get("count_distribution", "fixed"),
        mean_overlays=kwargs.get("mean_overlays"),
        geometric_p=kwargs.get("geometric_p"),
    )
    if num_overlays <= 0:
        return example

    width = int(example["width"])
    height = int(example["height"])

    base_masks = get_object_masks(example, allow_mask_load=True)
    if base_masks is None:
        base_masks = rectangular_masks_from_bboxes(example["objects"], width=width, height=height)
    set_object_masks(example, base_masks)
    focus_idx = compute_focus_index(example, **focus_kwargs)
    focus_idx = max(0, min(focus_idx, len(base_masks) - 1))
    base_video_ids = [record["video_id"] for record in example["objects"]]
    base_frame_indices = [record["frame_idx"] for record in example["objects"]]

    overlay_examples = context.sample_examples(
        num_overlays,
        exclude_current=kwargs.get("exclude_current", True),
        replace=kwargs.get("replace", False),
    )

    overlay_layers: list[dict[str, Any]] = []
    for overlay_example in overlay_examples:
        overlay_example = fit_example_to_canvas(overlay_example, width=width, height=height)

        overlay_scale_range = kwargs.get("scale_range")
        if overlay_scale_range is not None:
            overlay_example = apply_zoom(
                overlay_example,
                scale=random.uniform(*overlay_scale_range),
                fillcolor=kwargs.get("fillcolor", 0),
            )

        max_translate_frac = kwargs.get("max_translate_frac")
        max_translate_x_frac = kwargs.get("max_translate_x_frac", max_translate_frac)
        max_translate_y_frac = kwargs.get("max_translate_y_frac", max_translate_frac)
        dx = random_translate_delta(max_shift_frac=max_translate_x_frac, size=width)
        dy = random_translate_delta(max_shift_frac=max_translate_y_frac, size=height)
        overlay_example = apply_translation(
            overlay_example,
            dx=dx,
            dy=dy,
            fillcolor=kwargs.get("fillcolor", 0),
        )

        overlay_masks = get_object_masks(overlay_example, allow_mask_load=True)
        if overlay_masks is None:
            overlay_masks = rectangular_masks_from_bboxes(
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
                "categories": [int(record["category"]) for record in overlay_example["objects"]],
                "video_ids": [record["video_id"] for record in overlay_example["objects"]],
                "frame_indices": [record["frame_idx"] for record in overlay_example["objects"]],
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
    for index, (mask, category, video_id, frame_idx) in enumerate(
        zip(base_masks, [record["category"] for record in example["objects"]], base_video_ids, base_frame_indices)
    ):
        object_records.append(
            {
                "mask": np.asarray(mask, dtype=bool),
                "id": len(object_records),
                "category": int(category),
                "video_id": video_id,
                "frame_idx": frame_idx,
                "is_focus": index == focus_idx,
            }
        )
    for layer in kept_layers:
        for mask, category, video_id, frame_idx in zip(
            layer["masks"],
            layer["categories"],
            layer["video_ids"],
            layer["frame_indices"],
        ):
            object_records.append(
                {
                    "mask": np.asarray(mask, dtype=bool),
                    "id": len(object_records),
                    "category": int(category),
                    "video_id": video_id,
                    "frame_idx": frame_idx,
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
        component_count, largest_component_frac = count_connected_components(visible_mask)

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

    kept_records: list[dict[str, Any]] = []
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
            kept_records.append(
                {
                    "id": len(kept_records),
                    "category": record["category"],
                    "video_id": record["video_id"],
                    "frame_idx": record["frame_idx"],
                    "mask": visible_mask,
                }
            )

    example["image"] = Image.fromarray(composed_pixels)
    example["width"] = width
    example["height"] = height
    example["objects"] = kept_records
    if kept_records:
        kept_bboxes = np.asarray([mask2xywh(record["mask"]) for record in kept_records], dtype=np.float32)
        set_bboxes(example["objects"], kept_bboxes)
    return example
