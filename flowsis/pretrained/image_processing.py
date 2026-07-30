from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image


DetectorAnnotation = Mapping[str, Any]


def image_to_rgb_tensor(image: Any) -> torch.Tensor:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
        if not array.flags.writeable:
            array = array.copy()
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    if isinstance(image, np.ndarray):
        array = image if image.flags.writeable else image.copy()
        tensor = torch.from_numpy(np.ascontiguousarray(array))
    elif isinstance(image, torch.Tensor):
        tensor = image
    else:
        raise TypeError(
            "Device preprocessing expects PIL images, NumPy arrays, or tensors, "
            f"but received {type(image).__name__}."
        )

    if tensor.ndim != 3:
        raise ValueError(f"Expected a three-dimensional image, got {tuple(tensor.shape)}.")
    if tensor.shape[0] in (1, 3):
        return tensor
    if tensor.shape[-1] in (1, 3):
        return tensor.permute(2, 0, 1)
    raise ValueError(f"Could not identify image channels in shape {tuple(tensor.shape)}.")


def _resize_shape(height: int, width: int, image_size: int) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(height, width)}.")
    if height >= width:
        return image_size, max(1, round(image_size * width / height))
    return max(1, round(image_size * height / width)), image_size


def _prepare_detection_annotation(
    annotation: DetectorAnnotation,
    *,
    original_shape: tuple[int, int],
    resized_shape: tuple[int, int],
    padded_shape: tuple[int, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    original_height, original_width = original_shape
    resized_height, resized_width = resized_shape
    padded_height, padded_width = padded_shape
    objects = [
        obj
        for obj in annotation["annotations"]
        if int(obj.get("iscrowd", 0)) == 0
    ]

    boxes = torch.as_tensor(
        [obj["bbox"] for obj in objects],
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 4)
    classes = torch.as_tensor(
        [obj["category_id"] for obj in objects],
        dtype=torch.int64,
        device=device,
    )
    areas = torch.as_tensor(
        [obj["area"] for obj in objects],
        dtype=torch.float32,
        device=device,
    )
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(0, original_width)
    boxes[:, 1::2].clamp_(0, original_height)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[keep]
    classes = classes[keep]
    areas = areas[keep]

    width_scale = resized_width / original_width
    height_scale = resized_height / original_height
    boxes *= boxes.new_tensor([width_scale, height_scale, width_scale, height_scale])
    if boxes.numel():
        top_left = boxes[:, :2].clone()
        bottom_right = boxes[:, 2:]
        boxes[:, :2] = (top_left + bottom_right) * 0.5
        boxes[:, 2:] = bottom_right - top_left
        boxes /= boxes.new_tensor(
            [padded_width, padded_height, padded_width, padded_height]
        )

    return {
        "size": torch.tensor(padded_shape, dtype=torch.int64, device=device),
        "image_id": torch.tensor(
            [int(annotation.get("image_id", 0))],
            dtype=torch.int64,
            device=device,
        ),
        "class_labels": classes,
        "boxes": boxes,
        "area": areas * (width_scale * height_scale),
        "iscrowd": torch.zeros_like(classes),
        "orig_size": torch.tensor(original_shape, dtype=torch.int64, device=device),
    }


def _preprocess_pixel_batch(
    processor: Any,
    batch: torch.Tensor,
    *,
    image_size: int,
    device: torch.device,
    image_mean: torch.Tensor | None,
    image_std: torch.Tensor | None,
    pixel_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], tuple[int, int]]:
    if batch.ndim != 4 or batch.shape[1] not in (1, 3):
        raise ValueError(f"Expected a BCHW image batch, got {tuple(batch.shape)}.")

    original_shape = (int(batch.shape[-2]), int(batch.shape[-1]))
    resized_shape = _resize_shape(*original_shape, image_size)
    pixels = batch.to(device, non_blocking=True).float()
    if original_shape != resized_shape:
        pixels = F.interpolate(
            pixels,
            size=resized_shape,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pixels.round_().clamp_(0.0, 255.0)

    if getattr(processor, "do_rescale", True):
        pixels.mul_(float(getattr(processor, "rescale_factor", 1.0 / 255.0)))
    if getattr(processor, "do_normalize", False):
        if image_mean is None or image_std is None:
            image_mean = torch.as_tensor(processor.image_mean, device=device)
            image_std = torch.as_tensor(processor.image_std, device=device)
        pixels.sub_(image_mean.view(1, 3, 1, 1)).div_(
            image_std.view(1, 3, 1, 1)
        )

    resized_height, resized_width = resized_shape
    pixels = F.pad(
        pixels,
        (0, image_size - resized_width, 0, image_size - resized_height),
    )
    if pixel_mask is None:
        pixel_mask = torch.zeros(
            (1, image_size, image_size),
            dtype=torch.int64,
            device=device,
        )
        pixel_mask[:, :resized_height, :resized_width] = 1
    return (
        pixels,
        pixel_mask.expand(pixels.shape[0], -1, -1),
        original_shape,
        resized_shape,
    )


def preprocess_detr_images(
    processor: Any,
    images: Sequence[Any] | torch.Tensor,
    *,
    image_size: int,
    device: torch.device,
    annotations: Sequence[DetectorAnnotation] | None = None,
    image_mean: torch.Tensor | None = None,
    image_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]] | None]:
    """Resize, normalize, and pad a DETR batch on its execution device."""
    if isinstance(images, torch.Tensor):
        if images.ndim == 3:
            image_batches = [images.unsqueeze(0)]
        elif images.ndim == 4:
            if images.shape[1] not in (1, 3) and images.shape[-1] in (1, 3):
                images = images.permute(0, 3, 1, 2)
            image_batches = [images]
        else:
            raise ValueError(
                f"Expected a CHW or BCHW image tensor, got {tuple(images.shape)}."
            )
    else:
        image_batches = [
            image_to_rgb_tensor(image).unsqueeze(0)
            for image in images
        ]
    if not image_batches:
        raise ValueError("Expected at least one image.")
    batch_size = sum(batch.shape[0] for batch in image_batches)
    if annotations is not None and len(annotations) != batch_size:
        raise ValueError("The number of annotations must match the number of images.")

    processed_images: list[torch.Tensor] = []
    pixel_masks: list[torch.Tensor] = []
    processed_annotations: list[dict[str, torch.Tensor]] = []
    annotation_index = 0
    for image_batch in image_batches:
        pixels, mask, original_shape, resized_shape = _preprocess_pixel_batch(
            processor,
            image_batch,
            image_size=image_size,
            device=device,
            image_mean=image_mean,
            image_std=image_std,
        )
        processed_images.append(pixels)
        pixel_masks.append(mask)
        if annotations is not None:
            for annotation in annotations[
                annotation_index : annotation_index + image_batch.shape[0]
            ]:
                processed_annotations.append(
                    _prepare_detection_annotation(
                        annotation,
                        original_shape=original_shape,
                        resized_shape=resized_shape,
                        padded_shape=(image_size, image_size),
                        device=device,
                    )
                )
            annotation_index += image_batch.shape[0]

    return (
        torch.cat(processed_images),
        torch.cat(pixel_masks),
        processed_annotations if annotations is not None else None,
    )


def preprocess_detr_bgr_frame(
    processor: Any,
    frame_bgr: NDArray,
    *,
    image_size: int,
    device: torch.device,
    image_mean: torch.Tensor | None = None,
    image_std: torch.Tensor | None = None,
    pixel_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the numerical transforms shared by the supported DETR processors."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected an HWC BGR frame, got {frame_bgr.shape}.")
    if frame_bgr.shape[0] != frame_bgr.shape[1]:
        raise ValueError(
            "Device-side DETR preprocessing requires a square frame so it remains "
            "numerically equivalent to the reference processor."
        )
    batch = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(device)
    batch = batch.permute(2, 0, 1).flip(0).unsqueeze(0)
    pixels, mask, _, _ = _preprocess_pixel_batch(
        processor,
        batch,
        image_size=image_size,
        device=device,
        image_mean=image_mean,
        image_std=image_std,
        pixel_mask=pixel_mask,
    )
    return pixels, mask
