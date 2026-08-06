from __future__ import annotations

import torch
from transformers import AutoImageProcessor, DFineConfig, DFineForObjectDetection

from .detector import BaseDetector


class DFineDetector(BaseDetector):
    architecture = "dfine"
    expected_model_types = ("d_fine",)

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
    ) -> DFineDetector:
        config = DFineConfig.from_pretrained(
            model_name_or_path,
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
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        model = DFineForObjectDetection.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=num_labels is not None,
            local_files_only=local_files_only,
        )
        return cls(
            processor,
            model,
            source=source or model_name_or_path,
            image_size=image_size,
            device=device,
        )
