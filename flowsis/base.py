import torch.nn as nn
from pathlib import Path
from collections import deque
from typing import Mapping, Any
import numpy as np

from flowsis.pretrained import DetectorArchitecture, load_detector
from flowsis.selection import SelectionResult, select_first_detection, select_recurrent_detection
from flowsis.data import LabelPrompts

class FlowSISBase(nn.Module):
    def __init__(
        self,
        detector_name_or_path: str | None = None,
        head_path: str | Path | None = None,
        *,
        detector_architecture: DetectorArchitecture | None = None,
        detection_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.detector = load_detector(
            detector_name_or_path,
            architecture=detector_architecture,
        )
        self.id2label = self.detector.label_names
        self.prompt_cache: dict[str, torch.Tensor] = {}
        self.history: deque[Mapping[str, Any]] = deque(maxlen=args.history_size)
        self.previous_selection: SelectionResult | None = None
        self.detection_threshold = detection_threshold
        self.eval()

    def _select_detection(self) -> SelectionResult | None:
        if not self.history or len(self.history[-1]["scores"]) == 0:
            return None
        if self.previous_selection is None:
            return select_first_detection(self.history)
        return select_recurrent_detection(self.history, self.previous_selection)
    
    def infer(self, square_bgr):
        inference = self.detector.infer_frame(
            square_bgr,
            threshold=self.detection_threshold,
        )
        gpu_detections = inference.detections[0]
        detections = {
            key: value.detach().cpu().tolist()
            for key, value in gpu_detections.items()
        }
        self.history.append(detections)
        selection = self._select_detection()
        mask: np.ndarray | None = None
        selected_label: str | None = None
        if selection is not None:
            selected_label = self.id2label.get(selection.label, f"class_{selection.label}")
            text_embeddings = LabelPrompts.load_embeddings(
                selected_label,
                args.text_embeddings_dir,
                prompt_cache,
                device=device,
            )
            feature_maps = [
                feature.float() for feature in inference.feature_maps
            ]
            mask = predict_mask(
                head,
                feature_maps,
                text_embeddings,
                normalized_box(selection, width=width, height=height),
                output_size=(height, width),
                device=device,
                use_amp=args.amp,
                mask_threshold=args.mask_threshold,
            )
            previous_selection = selection