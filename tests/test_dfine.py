import numpy as np
import torch
from PIL import Image
from transformers import (
    DFineConfig,
    DFineForObjectDetection,
    RTDetrImageProcessor,
    ResNetConfig,
)

from flowsis.pretrained.dfine import DFineDetector


def build_tiny_dfine() -> DFineDetector:
    backbone = ResNetConfig(
        depths=[1, 1, 1, 1],
        hidden_sizes=[16, 32, 64, 128],
        embedding_size=8,
        out_features=["stage2", "stage3", "stage4"],
    )
    config = DFineConfig(
        backbone_config=backbone,
        encoder_in_channels=(32, 64, 128),
        encoder_hidden_dim=32,
        encoder_ffn_dim=64,
        encoder_attention_heads=4,
        d_model=32,
        decoder_in_channels=(32, 32, 32),
        decoder_ffn_dim=64,
        decoder_attention_heads=4,
        decoder_layers=2,
        num_queries=10,
        num_denoising=0,
        num_labels=3,
        max_num_bins=8,
        lqe_hidden_dim=16,
    )
    return DFineDetector(
        RTDetrImageProcessor(),
        DFineForObjectDetection(config),
        source="tiny-dfine",
        image_size=64,
    )


def test_dfine_inference_returns_detections_and_multiscale_features() -> None:
    detector = build_tiny_dfine()
    image = Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8))

    result = detector.infer(image)

    assert len(result.detections) == 1
    assert [tuple(feature.shape) for feature in result.feature_maps] == [
        (1, 32, 8, 8),
        (1, 32, 4, 4),
        (1, 32, 2, 2),
    ]
    assert result.query_embeddings is not None
    assert result.query_logits is not None
    assert result.query_boxes is not None
    assert result.query_embeddings.shape == (1, 10, 32)
    assert result.query_logits.shape == (1, 10, 3)
    assert result.query_boxes.shape == (1, 10, 4)
    query_indices = result.detections[0]["query_indices"]
    assert query_indices.shape == result.detections[0]["scores"].shape
    assert ((0 <= query_indices) & (query_indices < 10)).all()
    selected_boxes = result.query_boxes[0, query_indices]
    centers, sizes = selected_boxes.split(2, dim=-1)
    expected_boxes = torch.cat(
        (centers - 0.5 * sizes, centers + 0.5 * sizes),
        dim=-1,
    ) * torch.tensor([64.0, 48.0, 64.0, 48.0])
    torch.testing.assert_close(result.detections[0]["boxes"], expected_boxes)


def test_extract_feature_maps_supports_dfine() -> None:
    detector = build_tiny_dfine()
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))

    features = detector.extract_feature_maps([image])

    assert len(features) == 3
    assert all(feature.dtype == torch.float32 for feature in features)
    assert all(feature.ndim == 4 for feature in features)


def test_dfine_training_forward_returns_detection_losses() -> None:
    detector = build_tiny_dfine()
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    annotation = {
        "image_id": 1,
        "annotations": [
            {
                "bbox": [8.0, 8.0, 20.0, 20.0],
                "category_id": 1,
                "area": 400.0,
                "iscrowd": 0,
            }
        ],
    }

    result = detector([image], [annotation])

    assert result.loss is not None
    assert {"loss_bbox", "loss_giou", "loss_vfl"} <= result.loss_dict.keys()
