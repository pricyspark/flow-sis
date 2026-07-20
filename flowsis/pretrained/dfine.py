from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, DFineConfig, DFineForObjectDetection
from transformers.image_processing_utils import BaseImageProcessor

from .common import resolve_pretrained_source
from .rtdetrv2 import RTDetrV2


class DFine(RTDetrV2):
    """D-FINE backend with the detector and multiscale-feature API used by FlowSIS."""

    def __init__(
        self,
        processor: BaseImageProcessor,
        model: DFineForObjectDetection,
        device: str | torch.device | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.processor = processor
        self.model = model
        if device is not None:
            self.to(device)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "ustc-community/dfine-medium-obj365",
        *,
        cache_dir: str = "flowsis/models",
        num_labels: int | None = None,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        device: str | torch.device | None = None,
    ) -> DFine:
        resolved_source, local_files_only = resolve_pretrained_source(
            model_name_or_path,
            cache_dir,
        )
        config = DFineConfig.from_pretrained(
            resolved_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        if num_labels is not None:
            config.num_labels = int(num_labels)
            if id2label is None:
                id2label = {index: f"class_{index}" for index in range(num_labels)}
            if label2id is None:
                label2id = {label: index for index, label in id2label.items()}
            config.id2label = id2label
            config.label2id = label2id

        processor = AutoImageProcessor.from_pretrained(
            resolved_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        model = DFineForObjectDetection.from_pretrained(
            resolved_source,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=num_labels is not None,
            local_files_only=local_files_only,
        )
        return cls(processor, model, device)
