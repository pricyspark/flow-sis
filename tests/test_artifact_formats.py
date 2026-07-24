from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flowsis.data.features import load_feature_bundle, save_feature_bundle
from flowsis.head_checkpoint import load_head_checkpoint, save_head_checkpoint
from flowsis.utils.training import load_training_state, save_training_state


def test_feature_bundle_round_trip_and_validation(tmp_path: Path) -> None:
    feature_maps = [torch.ones(4, 3, 3), torch.ones(4, 2, 2)]
    save_feature_bundle(
        tmp_path,
        feature_maps,
        detector_architecture="dfine",
        detector_source="example/dfine",
        image_size=640,
    )

    bundle = load_feature_bundle(
        tmp_path,
        expected_levels=2,
        expected_channels=4,
        expected_image_size=640,
    )
    assert bundle.metadata.detector_architecture == "dfine"
    assert bundle.metadata.detector_source == "example/dfine"
    assert len(bundle.feature_maps) == 2

    with pytest.raises(ValueError, match="image size"):
        load_feature_bundle(tmp_path, expected_image_size=512)


def test_head_checkpoint_round_trip(tmp_path: Path) -> None:
    path = save_head_checkpoint(
        tmp_path,
        config={"image_dim": 4, "num_feature_levels": 2},
        state_dict={"weight": torch.ones(2, 2)},
    )
    checkpoint = load_head_checkpoint(path)
    assert checkpoint.config == {"image_dim": 4, "num_feature_levels": 2}
    assert torch.equal(checkpoint.state_dict["weight"], torch.ones(2, 2))


def test_training_state_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    save_training_state(
        tmp_path,
        optimizer,
        scheduler,
        epoch=3,
        global_step=17,
    )
    state = load_training_state(
        tmp_path,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert state.epoch == 3
    assert state.global_step == 17
