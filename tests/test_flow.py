import pytest
import torch
import torch.nn as nn

from flowsis.pretrained.flow import PTLFlowEstimator


class FakePTLFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.displacement = nn.Parameter(torch.tensor(0.0))
        self.last_images: torch.Tensor | None = None

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        images = inputs["images"]
        self.last_images = images
        batch_size, _, _, height, width = images.shape
        flows = self.displacement.expand(batch_size, 1, 2, height, width)
        return {"flows": flows}


def test_ptlflow_adapter_preserves_gradients_and_converts_rgb_to_bgr() -> None:
    backend = FakePTLFlow()
    estimator = PTLFlowEstimator(backend)
    current = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1).expand(1, 3, 4, 5)
    previous = current + 1.0

    flow = estimator(current, previous)

    assert flow.shape == (1, 2, 4, 5)
    assert backend.last_images is not None
    torch.testing.assert_close(
        backend.last_images[0, 0, :, 0, 0],
        torch.tensor([3.0, 2.0, 1.0]),
    )
    torch.testing.assert_close(
        backend.last_images[0, 1, :, 0, 0],
        torch.tensor([4.0, 3.0, 2.0]),
    )

    flow.sum().backward()
    assert backend.displacement.grad is not None


def test_ptlflow_adapter_rejects_nonstandard_output_shape() -> None:
    class BadPTLFlow(nn.Module):
        def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {"flows": torch.zeros(1, 3, 4, 5)}

    estimator = PTLFlowEstimator(BadPTLFlow())

    with pytest.raises(RuntimeError, match="Expected flow shape"):
        estimator(torch.rand(1, 3, 4, 5), torch.rand(1, 3, 4, 5))
