import torch
import torch.nn as nn
from pathlib import Path
from collections import deque
from typing import Mapping, Any, cast
import numpy as np

from flowsis.pretrained import DetectorArchitecture, load_detector
from flowsis.selection import SelectionResult, select_first_detection, select_recurrent_detection
from flowsis.head_checkpoint import load_head
from flowsis.data import LabelPrompts
from flowsis.utils import build_autocast_context

class FlowSISBase(nn.Module):
    def __init__(
        self,
        detector_name_or_path: str ,
        head_path: str | Path,
        text_embeddings_dir: str | Path,
        *,
        detector_architecture: DetectorArchitecture | None = None,
        detection_threshold: float = 0.5,
        history_size: int = 12,
    ) -> None:
        super().__init__()
        self.detector = load_detector(
            detector_name_or_path,
            architecture=detector_architecture,
        )
        self.text_embeddings_dir = Path(text_embeddings_dir)
        self.head, checkpont_path = load_head(head_path)
        self.id2label = self.detector.label_names
        self.prompt_cache: dict[str, torch.Tensor] = {}
        self.history: deque[Mapping[str, Any]] = deque(maxlen=history_size)
        self.previous_selection: SelectionResult | None = None
        self.detection_threshold = detection_threshold
        self.eval()

    def _select_detection(self) -> SelectionResult | None:
        if not self.history or len(self.history[-1]["scores"]) == 0:
            return None
        if self.previous_selection is None:
            return select_first_detection(self.history)
        return select_recurrent_detection(self.history, self.previous_selection)

    @torch.inference_mode()
    def predict_mask(
        self,
        feature_maps: list[torch.Tensor],
        text_embeddings: torch.Tensor,
        box: torch.Tensor,
        *,
        output_size: tuple[int, int],
        use_amp: bool,
        mask_threshold: float,
    ) -> np.ndarray:
        device = next(self.head.parameters()).device
        with build_autocast_context(enabled=use_amp, device=device):
            output = self.head(
                feature_maps,
                text_embeddings.unsqueeze(0),
                object_boxes=box.to(device),
                mask_output_size=output_size,
                return_intermediates=False,
            )
        # Threshold on the GPU so only one byte per pixel crosses PCIe instead of
        # a four-byte probability map. This produces the same binary mask used by
        # rendering and avoids CPU-side sigmoid and threshold work.
        mask = cast(torch.Tensor, output["mask_logits"])[0].sigmoid() >= mask_threshold
        return mask.to(dtype=torch.uint8).cpu().numpy()
    
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
            device = next(self.head.parameters()).device
            selected_label = self.id2label.get(selection.label, f"class_{selection.label}")
            text_embeddings = LabelPrompts.load_embeddings(
                selected_label,
                self.text_embeddings_dir,
                self.prompt_cache,
                device=device,
            )
            feature_maps = [
                feature.float() for feature in inference.feature_maps
            ]
            mask = self.predict_mask(
                feature_maps,
                text_embeddings,
                normalized_box(selection, width=width, height=height),
                output_size=(height, width),
                use_amp=args.amp,
                mask_threshold=args.mask_threshold,
            )
            previous_selection = selection
