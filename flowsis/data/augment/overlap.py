from __future__ import annotations

import random
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage

from ..masks import mask2xywh
from .common import (
    mask_union,
)

from flowsis.utils.common import init_rng

from ..classes import SampleContext


def _count_connected_components(
    mask: NDArray[np.bool_],
    connectivity: Literal[4, 8] = 8
) -> int:
    if connectivity == 4:
        structure = ndimage.generate_binary_structure(2, 1)
    elif connectivity == 8:
        structure = ndimage.generate_binary_structure(2, 2)
    else:
        raise ValueError("Connectivity must be 4 or 8")
    
    result = cast(
        tuple[NDArray[np.int_], int],
        ndimage.label(mask, structure=structure),
    )
    
    return result[1]

def _verify_overlap_example(
    example: dict[str, Any], 
    hard_threshold: float, 
    soft_threshold: float, 
    num_connected_components: int,
    connectivity: Literal[4, 8] = 8,
    masks: NDArray[np.bool_] | None = None,
    visible_masks: NDArray[np.bool_] | None = None,
) -> None:
    if hard_threshold > soft_threshold:
        raise ValueError("Hard threshold cannot be higher than soft threshold.")
    objects = example["objects"]
    if masks is None:
        masks = np.array([obj["mask"] for obj in objects], dtype=np.bool_)
    if visible_masks is None:
        visible_masks = np.array([obj["visible_mask"] for obj in objects], dtype=np.bool_)
    if len(objects) == 0:
        return
    if len(objects) != len(masks) or len(objects) != len(visible_masks):
        raise ValueError("Objects, masks, and visible_masks must have the same length.")
    if masks.ndim != 3 or visible_masks.ndim != 3:
        raise ValueError("Masks and visible_masks must be shaped as (num_objects, height, width).")

    visible_area = np.sum(visible_masks, axis=(1, 2), dtype=np.float32)
    total_area = np.sum(masks, axis=(1, 2), dtype=np.float32)
    visible_ratios = np.divide(
        visible_area,
        total_area,
        out=np.zeros(len(masks)),
        where=total_area > 0,
    )
    valid = visible_ratios >= soft_threshold
    maybe = (visible_ratios < soft_threshold) & (visible_ratios >= hard_threshold)
    
    for i, visible_mask in enumerate(visible_masks):
        if not maybe[i] or visible_area[i] == 0:
            continue
        
        connected_components = _count_connected_components(visible_mask, connectivity)
        if connected_components <= num_connected_components:
            valid[i] = True
            
    kept_objects = []
    for i, (v, obj) in enumerate(zip(valid, objects)):
        if not v:
            continue
        
        current_mask = visible_masks[i]
        bbox = mask2xywh(current_mask)
        assert bbox is not None
        obj["bbox"] = bbox
        obj["area"] = bbox[2] * bbox[3]
        obj["mask"] = current_mask
        #obj["visible_mask"] = current_mask
        obj.pop("visible_mask", None)
        obj["modified"] = True
        
        kept_objects.append(obj)
        
    example["objects"] = kept_objects
        
    


def overlap_augment(
    example: dict[str, Any], 
    *,
    context: SampleContext,
    **kwargs
) -> dict[str, Any]:
    rng = init_rng(kwargs.get("rng", None), kwargs.get("seed", None))
    
    if context is None:
        raise ValueError(
            "overlap_augment requires kwargs['augmentation_context']. "
            "Use it through TransformDataset/AugmentationPipeline with an indexable dataset."
        )
        
    img: Image.Image = example["image"]
    objects = example["objects"]
    height = example["height"]
    width = example["width"]
    
    base_masks = np.array([obj["mask"] for obj in objects], dtype=np.bool_)
        
    min_overlay = kwargs.get("min_overlays", 0)
    max_overlay = kwargs.get("max_overlays", min_overlay)
    p = kwargs.get("p", 0.5)
    if max_overlay < min_overlay:
        raise ValueError("max_overlays must be greater than or equal to min_overlays.")
    if not 0 <= p < 1:
        raise ValueError("p must satisfy 0 <= p < 1.")
    
    num_additional_samples = min(
        rng.geometric(1 - p) - 1,
        max_overlay - min_overlay
    )
    num_samples = min_overlay + num_additional_samples
    
    overlay_examples = context.sample_examples(num_samples, rng=rng)

    currently_blocked_mask = np.zeros((height, width), dtype=np.bool_)
    
    all_objects: list[dict[str, Any]] = []
    all_masks: list[NDArray[np.bool_]] = []
    all_visible_masks: list[NDArray[np.bool_]] = []
    
    for overlay_example in overlay_examples:
        overlay_img = overlay_example["image"]
        overlay_objects = overlay_example["objects"]
        overlay_masks = np.array([obj["mask"] for obj in overlay_objects], dtype=np.bool_)
        if overlay_img.width != width or overlay_img.height != height:
            raise ValueError("Overlay example image size must match the target canvas size.")
        if overlay_masks.ndim != 3:
            raise ValueError("Overlay masks must be shaped as (num_objects, height, width).")
        if overlay_masks.shape[1:] != (height, width):
            raise ValueError("Overlay masks must match the target canvas size.")
        
        currently_visible_mask = ~currently_blocked_mask
        visible_overlay_masks = overlay_masks & currently_visible_mask
        
        for obj, full_mask, visible_mask in zip(overlay_objects, overlay_masks, visible_overlay_masks):
            obj["visible_mask"] = visible_mask
            all_objects.append(obj)
            all_masks.append(full_mask)
            all_visible_masks.append(visible_mask)
    
        overlay_mask_union = mask_union(overlay_masks)
        currently_visible_overlay_mask = overlay_mask_union & currently_visible_mask
        mask_pil = Image.fromarray(
            currently_visible_overlay_mask.astype(np.uint8) * 255, 
            mode='L',
        )
        img.paste(overlay_img, (0, 0), mask_pil)
        currently_blocked_mask |= overlay_mask_union

    currently_visible_mask = ~currently_blocked_mask
    visible_base_masks = base_masks & currently_visible_mask
    for obj, full_mask, visible_mask in zip(objects, base_masks, visible_base_masks):
        obj["visible_mask"] = visible_mask
        all_objects.append(obj)
        all_masks.append(full_mask)
        all_visible_masks.append(visible_mask)

    example["objects"] = all_objects
      
    hard_threshold = kwargs.get("hard_threshold", 0.5)
    soft_threshold = kwargs.get("soft_threshold", 0.75)
    num_connected_components = kwargs.get("num_connected_components", 3)
    connectivity = kwargs.get("connectivity", 8)
        
    _verify_overlap_example(
        example,
        hard_threshold=hard_threshold,
        soft_threshold=soft_threshold,
        num_connected_components=num_connected_components,
        connectivity=connectivity,
        masks=np.asarray(all_masks, dtype=np.bool_),
        visible_masks=np.asarray(all_visible_masks, dtype=np.bool_),
    )
    
    return example
