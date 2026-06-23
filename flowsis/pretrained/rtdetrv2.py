import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from numpy.typing import NDArray
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, Iterable, cast
from transformers.image_utils import ImageInput
from transformers.utils.generic import ModelOutput
from transformers.feature_extraction_utils import BatchFeature
from transformers import RTDetrImageProcessor, RTDetrV2Config, RTDetrV2ForObjectDetection
from transformers.models.rt_detr_v2.modeling_rt_detr_v2 import RTDetrV2ObjectDetectionOutput

from .common import resolve_pretrained_source
from ..data.object_records import get_object_records


def _infer_single_image_size(image: Image.Image | NDArray | torch.Tensor) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        width, height = image.size
        return height, width
    if isinstance(image, np.ndarray):
        if image.ndim < 2:
            raise ValueError("Expected image array with at least 2 dimensions.")
        return int(image.shape[0]), int(image.shape[1])
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:
            return int(image.shape[0]), int(image.shape[1])
        if image.ndim == 3:
            if image.shape[0] in {1, 3}:
                return int(image.shape[1]), int(image.shape[2])
            return int(image.shape[0]), int(image.shape[1])
    raise TypeError(f"Unsupported image type: {type(image)!r}")


@dataclass
class RTDetrV2ForwardResult:
    loss: torch.Tensor | None
    loss_dict: dict[str, torch.Tensor]
    outputs: ModelOutput | None = None


@dataclass
class RTDetrV2InferenceResult:
    detections: list[dict[str, torch.Tensor]]
    encodings: Iterable[torch.Tensor]
    flat_encodings: dict[str, Any] | None


class RTDetrV2(nn.Module):
    """
    Thin RT-DETRv2 wrapper for training and inference.

    Inference returns:
    - detections: batch-aligned list of {"boxes", "scores", "labels"} in original image coordinates
    - encodings:
        - patch_tokens: [B, N, D] concatenated encoder feature tokens across all scales
        - image_embedding: [B, D] mean pooled from patch_tokens
        - feature_map_shapes: list[(H, W)] describing the encoder scales used to build patch_tokens
    """

    def __init__(
        self,
        processor: RTDetrImageProcessor,
        model: RTDetrV2ForObjectDetection,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()

        self.processor = processor
        self.model = model

        if device is not None:
            self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def preprocess(
        self,
        images: Image.Image | NDArray | torch.Tensor | Iterable[Image.Image | NDArray | torch.Tensor],
        annotations: Iterable[dict[str, Any]] | None = None,
        *,
        image_size: int | None = 640,
    ) -> BatchFeature:
        image_list = list(images) if isinstance(images, Iterable) else [images]
        image_input = cast(ImageInput, image_list)

        processor_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if image_size is not None:
            processor_kwargs["size"] = {
                "shortest_edge": image_size, 
                "longest_edge": image_size,
            }
            processor_kwargs["do_pad"] = True
            processor_kwargs["pad_size"] = {"height": image_size, "width": image_size}

        if annotations is None:
            batch = self.processor.preprocess(images=image_input, **processor_kwargs)
        else:
            batch = self.processor.preprocess(
                images=image_input,
                annotations=[
                    self._normalize_annotation(annotation) 
                    for annotation in annotations],
                **processor_kwargs,
            )
            
        batch = batch.to(self.device)
        if "labels" in batch:
            batch["labels"] = [
                {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in label.items()}
                for label in batch["labels"]
            ]
        
        return batch

    def forward(
        self,
        images: Image.Image | NDArray | torch.Tensor | Iterable[Image.Image | NDArray | torch.Tensor],
        annotations: Iterable[dict[str, Any]] | None = None,
        *,
        image_size: int | None = None,
        return_outputs: bool = False,
    ) -> RTDetrV2ForwardResult:
        batch = self.preprocess(images, annotations, image_size=image_size)
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch.get("pixel_mask"),
            labels=batch.get("labels"),
        )

        return RTDetrV2ForwardResult(
            loss=outputs.loss,
            loss_dict=dict(outputs.loss_dict or {}),
            outputs=outputs if return_outputs else None,
        )

    @torch.no_grad()
    def infer(
        self,
        images: Image.Image | NDArray | torch.Tensor | Iterable[Image.Image | NDArray | torch.Tensor],
        *,
        image_size: int | None = None,
        threshold: float = 0.1,
        flatten_outputs: bool = False,
    ) -> RTDetrV2InferenceResult:
        was_training = self.training
        self.eval()

        image_list = list(images) if isinstance(images, Iterable) else [images]
        batch = self.preprocess(image_list, annotations=None, image_size=image_size)
        outputs: RTDetrV2ObjectDetectionOutput = self.model(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch.get("pixel_mask"),
        )
        
        assert outputs.logits is not None
        target_sizes = torch.tensor(
            [_infer_single_image_size(image) for image in image_list],
            dtype=torch.int64,
            device=outputs.logits.device,
        )
        detections = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes, # type: ignore[arg-type] It's a HF typing bug
        )

        encodings = cast(list[torch.Tensor], outputs.encoder_last_hidden_state)
        
        if was_training:
            self.train()
        
        return RTDetrV2InferenceResult(
            detections=detections,
            encodings=encodings,
            flat_encodings=self._flatten_encodings(encodings) if flatten_outputs else None
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "PekingU/rtdetr_v2_r18vd",
        *,
        cache_dir: str = "flowsis/models",
        num_labels: int | None = None,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        device: str | torch.device | None = None,
    ) -> RTDetrV2:
        resolved_source, local_files_only = resolve_pretrained_source(
            model_name_or_path, 
            cache_dir,
        )
        config = RTDetrV2Config.from_pretrained(
            resolved_source, 
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        if num_labels is not None:
            config.num_labels = int(num_labels)
            if id2label is None:
                id2label = {index: f"class_{index}" for index in range(num_labels)}
                label2id = {label: index for index, label in id2label.items()}
            config.id2label = id2label
            config.label2id = label2id

        processor = RTDetrImageProcessor.from_pretrained(
            resolved_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        model = RTDetrV2ForObjectDetection.from_pretrained(
            resolved_source,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=num_labels is not None,
            local_files_only=local_files_only,
        )
        
        return cls(processor, model, device)

    def _normalize_annotation(self, annotation: dict[str, Any]) -> dict[str, Any]:
        # TODO: check for correctness
        if "annotations" in annotation:
            coco_annotations = annotation["annotations"]
        elif "objects" in annotation:
            coco_annotations = [
                {
                    "bbox": [float(value) for value in object_record["bbox"]],
                    "category_id": int(object_record["category"]),
                    "area": float(object_record["area"]),
                    "iscrowd": 0,
                }
                for object_record in get_object_records(annotation)
            ]
        else:
            raise ValueError("Expected annotation with either 'annotations' or 'objects'.")

        normalized_annotations = []
        for object_annotation in coco_annotations:
            normalized_annotations.append(
                {
                    **object_annotation,
                    "category_id": int(object_annotation["category_id"]),
                }
            )

        return {
            "image_id": int(annotation.get("image_id", 0)),
            "annotations": normalized_annotations,
        }

    def _flatten_encodings(self, encoder_feature_maps: Iterable[torch.Tensor]) -> dict[str, Any]:
        feature_maps = list(encoder_feature_maps)
        if not feature_maps:
            raise ValueError("RT-DETRv2 did not return encoder feature maps.")

        feature_map_shapes = [(int(feature_map.shape[-2]), int(feature_map.shape[-1])) for feature_map in feature_maps]
        patch_tokens = torch.cat(
            [feature_map.flatten(2).transpose(1, 2) for feature_map in feature_maps],
            dim=1,
        )
        image_embedding = patch_tokens.mean(dim=1)
        return {
            "patch_tokens": patch_tokens,
            "image_embedding": image_embedding,
            "feature_map_shapes": feature_map_shapes,
        }
