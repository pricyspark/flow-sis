import torch.nn as nn
from pathlib import Path
from collections import deque

from flowsis.pretrained import DetectorArchitecture, SigLIP2, load_detector


class FlowSISBase(nn.Module):
    def __init__(
        self,
        detector_name_or_path: str | None = None,
        head_path: str | Path | None = None,
        *,
        detector_architecture: DetectorArchitecture | None = None,
        siglip2_name_or_path: str = "google/siglip2-base-patch16-224",
    ) -> None:
        super().__init__()
        self.detector = load_detector(
            detector_name_or_path,
            architecture=detector_architecture,
        )
        self.id2label = self.detector.label_names
        prompt_cache: dict[str, torch.Tensor] = {}
        history: deque[Mapping[str, Any]] = deque(maxlen=args.history_size)
        previous_selection: SelectionResult | None = None

        self.eval()

    def infer(self, square_bgr):
        inference = self.detector.infer_frame(square_bgr)
