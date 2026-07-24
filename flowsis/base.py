import torch.nn as nn

from flowsis.pretrained import DetectorArchitecture, SigLIP2, load_detector


class FlowSISBase(nn.Module):
    def __init__(
        self,
        detector_name_or_path: str | None = None,
        *,
        detector_architecture: DetectorArchitecture | None = None,
        siglip2_name_or_path: str = "google/siglip2-base-patch16-224",
    ) -> None:
        super().__init__()
        self.detector = load_detector(
            detector_name_or_path,
            architecture=detector_architecture,
        )
        self.siglip2 = SigLIP2.from_pretrained(siglip2_name_or_path)
