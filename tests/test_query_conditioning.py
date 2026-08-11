import pytest
import torch

from flowsis.base_head import BaseFusionHead
from flowsis.cli.train.train_base_head import compute_batch_loss, match_object_queries
from flowsis.pretrained import DetectorInferenceResult


def build_query_head() -> BaseFusionHead:
    return BaseFusionHead(
        num_decode_layers=0,
        decode_embed_dim=4,
        image_dim=4,
        text_dim=4,
        nhead=1,
        decode_ffn_dim=8,
        dropout=0.0,
        activation="gelu",
        num_feature_levels=1,
        decode_pos_encode="none",
        image_self_attention="none",
        decode_window_size=2,
        use_shifted_windows=False,
        multiscale_merge="conv",
        deformable_num_points=2,
        deformable_offset_scale=1.0,
        query_dim=4,
        mask_upsample_scales=(1,),
    )


def test_query_conditioning_starts_as_identity_and_is_trainable() -> None:
    head = build_query_head()
    features = [torch.randn(2, 4, 3, 3)]
    queries = torch.tensor(
        [[1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]
    )

    conditioned = head._condition_on_object_query(features, queries)
    torch.testing.assert_close(conditioned[0], features[0])

    assert head.query_affine is not None
    with torch.no_grad():
        head.query_affine.weight[4, 0] = 1.0
    conditioned = head._condition_on_object_query(features, queries)

    assert not torch.allclose(conditioned[0][0], features[0][0])
    assert not torch.allclose(conditioned[0][0], conditioned[0][1])


def test_query_conditioned_head_requires_queries() -> None:
    head = build_query_head()

    with pytest.raises(ValueError, match="no object_queries"):
        head(
            [torch.randn(1, 4, 3, 3)],
            torch.randn(1, 2, 4),
        )


def test_query_matching_uses_class_box_cost_and_one_to_one_assignment() -> None:
    query_logits = torch.tensor(
        [[[8.0, -8.0], [-8.0, 8.0], [7.0, -7.0]]]
    )
    query_boxes = torch.tensor(
        [
            [
                [0.25, 0.25, 0.2, 0.2],
                [0.75, 0.75, 0.2, 0.2],
                [0.80, 0.20, 0.2, 0.2],
            ]
        ]
    )
    object_image_indices = torch.tensor([0, 0])
    object_labels = torch.tensor([0, 1])
    object_boxes = torch.tensor(
        [[0.15, 0.15, 0.35, 0.35], [0.65, 0.65, 0.85, 0.85]]
    )

    matches = match_object_queries(
        query_logits,
        query_boxes,
        object_image_indices,
        object_labels,
        object_boxes,
    )

    assert matches.tolist() == [0, 1]


def test_online_batch_routes_matched_queries_into_the_head() -> None:
    class FakeOnlineDetector:
        def infer(self, images, *, threshold, device_preprocess):
            return DetectorInferenceResult(
                detections=[],
                feature_maps=(torch.randn(1, 4, 4, 4),),
                query_embeddings=torch.tensor(
                    [[[1.0, -1.0, 0.5, 0.0], [-1.0, 1.0, 0.0, 0.5]]]
                ),
                query_logits=torch.tensor([[[8.0, -8.0], [-8.0, 8.0]]]),
                query_boxes=torch.tensor(
                    [[[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]]]
                ),
            )

    head = build_query_head()
    batch = {
        "images": torch.zeros(1, 3, 8, 8),
        "object_image_indices": torch.tensor([0]),
        "object_labels": torch.tensor([0]),
        "object_boxes": torch.tensor([[0.15, 0.15, 0.35, 0.35]]),
        "text_embeddings": torch.randn(1, 2, 4),
        "target_masks": torch.zeros(1, 8, 8),
    }

    result = compute_batch_loss(
        head,
        batch,
        online_encoder=FakeOnlineDetector(),
        use_amp=False,
        bce_weight=1.0,
        dice_weight=1.0,
        dice_smooth=1.0,
        prompt_dropout=0.0,
    )
    result["loss"].backward()

    assert head.query_affine is not None
    assert head.query_affine.weight.grad is not None
    assert head.query_affine.weight.grad.abs().sum() > 0
