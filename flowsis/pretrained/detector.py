from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, cast

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.utils.generic import ModelOutput

from flowsis.artifacts import atomic_write_text
from flowsis.data.object_records import get_object_records

from .common import resolve_pretrained_source
from .image_processing import preprocess_detr_bgr_frame, preprocess_detr_images


DetectorArchitecture = Literal["rtdetrv2", "dfine"]
DetectorImage = Image.Image | NDArray | torch.Tensor
DetectorImages = DetectorImage | Iterable[DetectorImage]
Detection = dict[str, torch.Tensor]

DETECTOR_METADATA_FILE = "flowsis_detector.json"
DETECTOR_METADATA_VERSION = 1


@dataclass(frozen=True)
class DetectorSpec:
    architecture: DetectorArchitecture
    adapter_module: str
    adapter_class: str
    default_model: str
    default_output_dir: Path
    model_types: tuple[str, ...]

    def adapter_type(self) -> type[BaseDetector]:
        module = import_module(self.adapter_module)
        adapter = getattr(module, self.adapter_class)
        if not isinstance(adapter, type) or not issubclass(adapter, BaseDetector):
            raise TypeError(
                f"{self.adapter_module}.{self.adapter_class} is not a detector adapter."
            )
        return adapter


DETECTOR_SPECS: dict[DetectorArchitecture, DetectorSpec] = {
    "rtdetrv2": DetectorSpec(
        architecture="rtdetrv2",
        adapter_module="flowsis.pretrained.rtdetrv2",
        adapter_class="RTDetrV2Detector",
        default_model="PekingU/rtdetr_v2_r50vd",
        default_output_dir=Path("outputs/detectors/rtdetrv2"),
        model_types=("rt_detr_v2",),
    ),
    "dfine": DetectorSpec(
        architecture="dfine",
        adapter_module="flowsis.pretrained.dfine",
        adapter_class="DFineDetector",
        default_model="ustc-community/dfine-medium-obj365",
        default_output_dir=Path("outputs/detectors/dfine"),
        model_types=("d_fine",),
    ),
}
DETECTOR_ARCHITECTURES: tuple[DetectorArchitecture, ...] = tuple(DETECTOR_SPECS)


@dataclass
class DetectorForwardResult:
    loss: torch.Tensor | None
    loss_dict: dict[str, torch.Tensor]


@dataclass
class DetectorInferenceResult:
    detections: list[Detection]
    feature_maps: tuple[torch.Tensor, ...]


class Detector(Protocol):
    architecture: DetectorArchitecture
    source: str

    @property
    def device(self) -> torch.device: ...

    @property
    def image_size(self) -> int: ...

    @property
    def label_names(self) -> dict[int, str]: ...

    @property
    def model_config(self) -> dict[str, Any]: ...

    def __call__(
        self,
        images: DetectorImages,
        annotations: Iterable[dict[str, Any]] | None = None,
        *,
        device_preprocess: bool = False,
    ) -> DetectorForwardResult: ...

    def infer(
        self,
        images: DetectorImages,
        *,
        threshold: float = 0.1,
        device_preprocess: bool | None = None,
    ) -> DetectorInferenceResult: ...

    def infer_frame(
        self,
        frame_bgr: NDArray,
        *,
        threshold: float = 0.1,
        device_preprocess: bool = True,
    ) -> DetectorInferenceResult: ...

    def extract_feature_maps(
        self,
        images: DetectorImages,
        *,
        device_preprocess: bool = False,
    ) -> tuple[torch.Tensor, ...]: ...

    def split_backbone_parameters(
        self,
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]: ...

    def train(self, mode: bool = True) -> Detector: ...
    def eval(self) -> Detector: ...
    def parameters(self, recurse: bool = True) -> Iterable[nn.Parameter]: ...
    def requires_grad_(self, requires_grad: bool = True) -> Detector: ...
    def save_pretrained(self, output_dir: str | Path) -> None: ...


class BaseDetector(nn.Module):
    """Backend-neutral adapter for DETR-family models used by FlowSIS."""

    architecture: ClassVar[DetectorArchitecture]
    expected_model_types: ClassVar[tuple[str, ...]]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        source: str | None = None,
        cache_dir: str = "flowsis/models",
        local_files_only: bool = False,
        num_labels: int | None = None,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        image_size: int = 640,
        device: str | torch.device | None = None,
    ) -> BaseDetector:
        raise NotImplementedError

    def __init__(
        self,
        processor: Any,
        model: nn.Module,
        *,
        source: str,
        image_size: int = 640,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        self._processor = processor
        self._model = model
        self.source = source
        self._configured_image_size = image_size
        self.register_buffer(
            "_preprocess_mean",
            torch.as_tensor(
                getattr(processor, "image_mean", (0.0, 0.0, 0.0)),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_preprocess_std",
            torch.as_tensor(
                getattr(processor, "image_std", (1.0, 1.0, 1.0)),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_full_pixel_mask",
            torch.ones((1, image_size, image_size), dtype=torch.int64),
            persistent=False,
        )
        self._validate_model_type()
        if device is not None:
            self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self._model.parameters()).device

    @property
    def image_size(self) -> int:
        return self._configured_image_size

    @property
    def label_names(self) -> dict[int, str]:
        config = getattr(self._model, "config", None)
        raw = getattr(config, "id2label", None)
        if isinstance(raw, Mapping):
            labels = {
                int(index): str(label) if label else f"class_{int(index)}"
                for index, label in raw.items()
            }
            if labels:
                return labels
        num_labels = int(getattr(config, "num_labels", 0) or 0)
        return {index: f"class_{index}" for index in range(num_labels)}

    @property
    def model_config(self) -> dict[str, Any]:
        config = getattr(self._model, "config", None)
        if config is None or not hasattr(config, "to_dict"):
            raise TypeError("Detector model does not expose a serializable configuration.")
        return cast(dict[str, Any], config.to_dict())

    def _validate_model_type(self) -> None:
        model_type = getattr(getattr(self._model, "config", None), "model_type", None)
        if model_type not in self.expected_model_types:
            expected = ", ".join(self.expected_model_types)
            raise ValueError(
                f"Checkpoint model_type {model_type!r} is incompatible with detector "
                f"architecture {self.architecture!r}; expected one of: {expected}."
            )

    @staticmethod
    def _image_list(images: DetectorImages) -> list[DetectorImage]:
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            return list(images)
        if isinstance(images, (Image.Image, np.ndarray, torch.Tensor)):
            return [images]
        return list(images)

    @staticmethod
    def _original_image_size(image: DetectorImage) -> tuple[int, int]:
        if isinstance(image, Image.Image):
            width, height = image.size
            return height, width
        if isinstance(image, np.ndarray):
            if image.ndim < 2:
                raise ValueError("Expected image array with at least two dimensions.")
            return int(image.shape[0]), int(image.shape[1])
        if image.ndim == 2:
            return int(image.shape[0]), int(image.shape[1])
        if image.ndim == 3:
            if image.shape[0] in {1, 3}:
                return int(image.shape[1]), int(image.shape[2])
            return int(image.shape[0]), int(image.shape[1])
        raise ValueError(f"Expected a two- or three-dimensional image tensor, got {image.shape}.")

    @staticmethod
    def _normalize_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
        if "annotations" in annotation:
            objects = annotation["annotations"]
        elif "objects" in annotation:
            objects = [
                {
                    "bbox": [float(value) for value in record["bbox"]],
                    "category_id": int(record["category"]),
                    "area": float(record["area"]),
                    "iscrowd": 0,
                }
                for record in get_object_records(annotation)
            ]
        else:
            raise ValueError("Expected annotation with either 'annotations' or 'objects'.")

        return {
            "image_id": int(annotation.get("image_id", 0)),
            "annotations": [
                {**obj, "category_id": int(obj["category_id"])} for obj in objects
            ],
        }

    def preprocess(
        self,
        images: DetectorImages,
        annotations: Iterable[dict[str, Any]] | None = None,
        *,
        device_preprocess: bool = False,
    ) -> BatchFeature:
        if device_preprocess:
            image_values: Any = images
            if not isinstance(images, torch.Tensor):
                image_values = self._image_list(images)
            annotation_values = (
                None
                if annotations is None
                else [
                    self._normalize_annotation(annotation)
                    for annotation in annotations
                ]
            )
            pixel_values, pixel_mask, labels = preprocess_detr_images(
                self._processor,
                image_values,
                image_size=self.image_size,
                device=self.device,
                annotations=annotation_values,
                image_mean=self._preprocess_mean,
                image_std=self._preprocess_std,
            )
            batch = BatchFeature(
                {
                    "pixel_values": pixel_values,
                    "pixel_mask": pixel_mask,
                }
            )
            if labels is not None:
                batch["labels"] = labels
            return batch

        image_list = self._image_list(images)
        processor_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "size": {
                "shortest_edge": self.image_size,
                "longest_edge": self.image_size,
            },
            "do_pad": True,
            "pad_size": {
                "height": self.image_size,
                "width": self.image_size,
            },
        }

        image_input = cast(ImageInput, image_list)
        if annotations is None:
            batch = self._processor.preprocess(images=image_input, **processor_kwargs)
        else:
            batch = self._processor.preprocess(
                images=image_input,
                annotations=[
                    self._normalize_annotation(annotation) for annotation in annotations
                ],
                **processor_kwargs,
            )

        batch = batch.to(self.device)
        if "labels" in batch:
            batch["labels"] = [
                {
                    key: value.to(self.device) if torch.is_tensor(value) else value
                    for key, value in label.items()
                }
                for label in batch["labels"]
            ]
        return batch

    def _forward_model(self, batch: Mapping[str, Any]) -> ModelOutput:
        return self._model(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch.get("pixel_mask"),
            labels=batch.get("labels"),
        )

    def _inference_model(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: torch.Tensor | None,
    ) -> ModelOutput:
        return self._model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def _extract_feature_maps(self, outputs: ModelOutput) -> tuple[torch.Tensor, ...]:
        raw = getattr(outputs, "encoder_last_hidden_state", None)
        if not isinstance(raw, (list, tuple)) or not raw:
            raise RuntimeError(
                f"{self.architecture} did not return projected multiscale encoder maps."
            )
        maps = tuple(raw)
        if any(not isinstance(feature, torch.Tensor) or feature.ndim != 4 for feature in maps):
            shapes = [getattr(feature, "shape", None) for feature in maps]
            raise RuntimeError(
                f"Expected detector feature maps shaped [B,C,H,W], got {shapes}."
            )
        return maps

    def forward(
        self,
        images: DetectorImages,
        annotations: Iterable[dict[str, Any]] | None = None,
        *,
        device_preprocess: bool = False,
    ) -> DetectorForwardResult:
        outputs = self._forward_model(
            self.preprocess(
                images,
                annotations,
                device_preprocess=device_preprocess,
            )
        )
        return DetectorForwardResult(
            loss=getattr(outputs, "loss", None),
            loss_dict=dict(getattr(outputs, "loss_dict", None) or {}),
        )

    def _postprocess(
        self,
        outputs: ModelOutput,
        *,
        original_sizes: list[tuple[int, int]],
        threshold: float,
    ) -> DetectorInferenceResult:
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError(f"{self.architecture} inference did not return logits.")
        target_sizes = torch.tensor(
            original_sizes,
            dtype=torch.int64,
            device=logits.device,
        )
        detections = self._processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )
        feature_maps = self._extract_feature_maps(outputs)
        return DetectorInferenceResult(
            detections=detections,
            feature_maps=feature_maps,
        )

    @torch.inference_mode()
    def infer(
        self,
        images: DetectorImages,
        *,
        threshold: float = 0.1,
        device_preprocess: bool | None = None,
    ) -> DetectorInferenceResult:
        was_training = self.training
        self.eval()
        try:
            image_list = self._image_list(images)
            if device_preprocess is None:
                device_preprocess = self.device.type != "cpu"
            preprocess_input: DetectorImages = image_list
            if isinstance(images, torch.Tensor) and images.ndim == 4:
                preprocess_input = images
            batch = self.preprocess(
                preprocess_input,
                device_preprocess=device_preprocess,
            )
            outputs = self._inference_model(
                batch["pixel_values"],
                batch.get("pixel_mask"),
            )
            return self._postprocess(
                outputs,
                original_sizes=[
                    self._original_image_size(image) for image in image_list
                ],
                threshold=threshold,
            )
        finally:
            self.train(was_training)

    @torch.inference_mode()
    def infer_preprocessed(
        self,
        pixel_values: torch.Tensor,
        *,
        original_sizes: list[tuple[int, int]],
        threshold: float = 0.1,
        pixel_mask: torch.Tensor | None = None,
    ) -> DetectorInferenceResult:
        if pixel_values.ndim != 4:
            raise ValueError(
                f"Expected pixel_values shaped [B,C,H,W], got {tuple(pixel_values.shape)}."
            )
        if len(original_sizes) != pixel_values.shape[0]:
            raise ValueError("original_sizes must contain one entry per image.")
        was_training = self.training
        self.eval()
        try:
            outputs = self._inference_model(
                pixel_values.to(self.device),
                None if pixel_mask is None else pixel_mask.to(self.device),
            )
            return self._postprocess(
                outputs,
                original_sizes=original_sizes,
                threshold=threshold,
            )
        finally:
            self.train(was_training)

    def preprocess_bgr_frame(
        self,
        frame_bgr: NDArray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return preprocess_detr_bgr_frame(
            self._processor,
            frame_bgr,
            image_size=self.image_size,
            device=self.device,
            image_mean=self._preprocess_mean,
            image_std=self._preprocess_std,
            pixel_mask=self._full_pixel_mask,
        )

    def infer_frame(
        self,
        frame_bgr: NDArray,
        *,
        threshold: float = 0.1,
        device_preprocess: bool = True,
    ) -> DetectorInferenceResult:
        height, width = frame_bgr.shape[:2]
        if not device_preprocess:
            return self.infer(
                frame_bgr[..., ::-1].copy(),
                threshold=threshold,
            )
        pixels, mask = self.preprocess_bgr_frame(frame_bgr)
        return self.infer_preprocessed(
            pixels,
            pixel_mask=mask,
            original_sizes=[(height, width)],
            threshold=threshold,
        )

    @torch.inference_mode()
    def extract_feature_maps(
        self,
        images: DetectorImages,
        *,
        device_preprocess: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        batch = self.preprocess(
            images,
            device_preprocess=device_preprocess,
        )
        outputs = self._inference_model(
            batch["pixel_values"],
            batch.get("pixel_mask"),
        )
        return tuple(feature.float() for feature in self._extract_feature_maps(outputs))

    def split_backbone_parameters(
        self,
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        backbone: list[nn.Parameter] = []
        other: list[nn.Parameter] = []
        for name, parameter in self._model.named_parameters():
            if not parameter.requires_grad:
                continue
            target = backbone if "backbone" in name.split(".") else other
            target.append(parameter)
        return backbone, other

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(output_path)
        self._processor.save_pretrained(output_path)
        metadata = {
            "format_version": DETECTOR_METADATA_VERSION,
            "architecture": self.architecture,
            "source": self.source,
            "image_size": self.image_size,
            "model_type": self.model_config.get("model_type"),
        }
        atomic_write_text(
            output_path / DETECTOR_METADATA_FILE,
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )


def get_detector_spec(architecture: DetectorArchitecture) -> DetectorSpec:
    try:
        return DETECTOR_SPECS[architecture]
    except KeyError as exc:
        raise ValueError(f"Unsupported detector architecture: {architecture!r}") from exc


def detector_default_model(architecture: DetectorArchitecture) -> str:
    return get_detector_spec(architecture).default_model


def detector_default_output_dir(architecture: DetectorArchitecture) -> Path:
    return get_detector_spec(architecture).default_output_dir


def _metadata_architecture(path: Path) -> DetectorArchitecture | None:
    metadata_path = path / DETECTOR_METADATA_FILE
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format_version") != DETECTOR_METADATA_VERSION:
        raise ValueError(
            f"Unsupported detector metadata version in {metadata_path}: "
            f"{metadata.get('format_version')!r}."
        )
    architecture = metadata.get("architecture")
    if architecture not in DETECTOR_SPECS:
        raise ValueError(f"Unknown detector architecture in {metadata_path}: {architecture!r}.")
    typed_architecture = cast(DetectorArchitecture, architecture)
    model_type = metadata.get("model_type")
    if model_type not in get_detector_spec(typed_architecture).model_types:
        raise ValueError(
            f"Detector metadata {metadata_path} has incompatible model_type "
            f"{model_type!r}."
        )
    return typed_architecture


def infer_detector_architecture(
    model_name_or_path: str | Path,
    *,
    cache_dir: str = "flowsis/models",
) -> DetectorArchitecture:
    source = str(model_name_or_path)
    path = Path(source)
    if path.is_dir() and (architecture := _metadata_architecture(path)) is not None:
        return architecture

    resolved_source, local_files_only = resolve_pretrained_source(source)
    config = AutoConfig.from_pretrained(
        resolved_source,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model_type = getattr(config, "model_type", None)
    matches = [
        architecture
        for architecture, spec in DETECTOR_SPECS.items()
        if model_type in spec.model_types
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not infer a supported detector architecture from model_type "
            f"{model_type!r} at {source!r}."
        )
    return matches[0]


def resolve_detector(
    model_name_or_path: str | Path | None = None,
    *,
    architecture: DetectorArchitecture | None = None,
    cache_dir: str = "flowsis/models",
) -> tuple[DetectorSpec, str]:
    if architecture is None and model_name_or_path is None:
        architecture = "rtdetrv2"
    if architecture is None:
        architecture = infer_detector_architecture(
            cast(str | Path, model_name_or_path),
            cache_dir=cache_dir,
        )
    spec = get_detector_spec(architecture)
    source = spec.default_model if model_name_or_path is None else str(model_name_or_path)

    path = Path(source)
    metadata_architecture = _metadata_architecture(path) if path.is_dir() else None
    if metadata_architecture is not None and metadata_architecture != architecture:
        raise ValueError(
            f"Checkpoint {source!r} contains {metadata_architecture!r}, not "
            f"the requested {architecture!r} detector."
        )
    if metadata_architecture is None:
        resolved_source, local_files_only = resolve_pretrained_source(source)
        config = AutoConfig.from_pretrained(
            resolved_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        model_type = getattr(config, "model_type", None)
        if model_type not in spec.model_types:
            raise ValueError(
                f"Checkpoint {source!r} has model_type {model_type!r}, which is "
                f"incompatible with detector architecture {architecture!r}."
            )
    return spec, source


def load_detector(
    model_name_or_path: str | Path | None = None,
    *,
    architecture: DetectorArchitecture | None = None,
    cache_dir: str = "flowsis/models",
    num_labels: int | None = None,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    image_size: int | None = None,
    device: str | torch.device | None = None,
) -> Detector:
    spec, source = resolve_detector(
        model_name_or_path,
        architecture=architecture,
        cache_dir=cache_dir,
    )
    resolved_source, local_files_only = resolve_pretrained_source(source)
    if image_size is None:
        metadata_path = Path(source) / DETECTOR_METADATA_FILE
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            image_size = int(metadata.get("image_size", 640))
        else:
            image_size = 640
    return spec.adapter_type().from_pretrained(
        resolved_source,
        source=source,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        image_size=image_size,
        device=device,
    )
