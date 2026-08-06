import pytest
import torch

from flowsis.temporal import (
    TemporalRefinementBranch,
    TemporalState,
    warp_backward,
)


def test_zero_flow_is_identity() -> None:
    values = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    flow = torch.zeros(1, 2, 3, 4)

    warped, validity = warp_backward(values, flow)

    torch.testing.assert_close(warped, values)
    assert validity.shape == (1, 1, 3, 4)
    assert validity.all()


def test_backward_flow_samples_source_coordinates_and_marks_bounds() -> None:
    values = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
    flow = torch.zeros(1, 2, 1, 4)
    flow[:, 0] = 1.0

    warped, validity = warp_backward(values, flow)

    torch.testing.assert_close(warped, torch.tensor([[[[1.0, 2.0, 3.0, 0.0]]]]))
    torch.testing.assert_close(
        validity,
        torch.tensor([[[[1.0, 1.0, 1.0, 0.0]]]]),
    )


def test_temporal_branch_returns_full_resolution_diagnostics() -> None:
    branch = TemporalRefinementBranch(channels=(8, 16))
    current_frame = torch.rand(2, 3, 16, 20)
    previous_frame = torch.rand(2, 3, 16, 20)
    base_logits = torch.randn(2, 1, 16, 20, requires_grad=True)
    previous_logits = torch.randn(2, 1, 16, 20, requires_grad=True)
    flow = torch.zeros(2, 2, 16, 20, requires_grad=True)

    output = branch(
        current_frame,
        previous_frame,
        base_logits,
        previous_logits,
        flow,
    )

    for tensor in (
        output.final_logits,
        output.propagated_logits,
        output.propagation_gate,
        output.residual_gate,
        output.logit_residual,
        output.warp_validity,
        output.photometric_residual,
    ):
        assert tensor.shape == (2, 1, 16, 20)

    output.final_logits.mean().backward()
    assert base_logits.grad is not None
    assert previous_logits.grad is not None
    assert flow.grad is not None
    assert branch.gate_head.weight.grad is not None
    assert branch.residual_head.weight.grad is not None


def test_invalid_warp_falls_back_to_base_at_initialization() -> None:
    branch = TemporalRefinementBranch(channels=(8,))
    current_frame = torch.rand(1, 3, 8, 8)
    previous_frame = torch.rand(1, 3, 8, 8)
    base_logits = torch.randn(1, 1, 8, 8)
    previous_logits = torch.randn(1, 1, 8, 8)
    flow = torch.full((1, 2, 8, 8), 100.0)

    output = branch(
        current_frame,
        previous_frame,
        base_logits,
        previous_logits,
        flow,
    )

    assert not output.warp_validity.any()
    torch.testing.assert_close(output.final_logits, base_logits)


def test_temporal_state_detaches_recurrent_tensors() -> None:
    branch = TemporalRefinementBranch(channels=(8,))
    frame = torch.rand(1, 3, 8, 8, requires_grad=True)
    base_logits = torch.randn(1, 1, 8, 8, requires_grad=True)
    output = branch(
        frame,
        torch.rand_like(frame),
        base_logits,
        torch.randn_like(base_logits),
        torch.zeros(1, 2, 8, 8),
    )

    state = TemporalState.from_output(frame, output, identity=7)

    assert not state.frame.requires_grad
    assert not state.logits.requires_grad
    assert state.identity == 7


def test_temporal_branch_rejects_incompatible_shapes() -> None:
    branch = TemporalRefinementBranch(channels=(8,))

    with pytest.raises(ValueError, match="current_base_logits"):
        branch(
            torch.rand(1, 3, 8, 8),
            torch.rand(1, 3, 8, 8),
            torch.rand(1, 8, 8),
            torch.rand(1, 1, 8, 8),
            torch.zeros(1, 2, 8, 8),
        )
