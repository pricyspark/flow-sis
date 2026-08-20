import numpy as np
import torch
from PIL import Image

from flowsis.cli.eval.ablate_text_conditioning import (
    ablate_embeddings,
    make_wrong_label_map,
    summarize,
)
from flowsis.cli.train.train_base_head import collate_online_examples


def test_wrong_label_map_is_reproducible_derangement() -> None:
    first = make_wrong_label_map([0, 1, 2, 3], seed=7)
    second = make_wrong_label_map([3, 2, 1, 0], seed=7)

    assert first == second
    assert set(first) == set(first.values()) == {0, 1, 2, 3}
    assert all(source != target for source, target in first.items())


def test_embedding_ablation_keeps_shape_and_swaps_labels() -> None:
    correct = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    catalog = {0: correct[0], 1: correct[1]}
    labels = torch.tensor([0, 1])
    permutation = torch.tensor([2, 0, 1])
    kwargs = {
        "label_embeddings": catalog,
        "wrong_label_map": {0: 1, 1: 0},
        "feature_permutation": permutation,
    }

    wrong = ablate_embeddings("wrong_label", correct, labels, **kwargs)
    scrambled = ablate_embeddings("scrambled", correct, labels, **kwargs)
    zero = ablate_embeddings("zero", correct, labels, **kwargs)

    torch.testing.assert_close(wrong, correct.flip(0))
    torch.testing.assert_close(scrambled, correct[..., permutation])
    torch.testing.assert_close(zero, torch.zeros_like(correct))


def test_summary_uses_paired_changes() -> None:
    correct = {
        name: np.array([0.2, 0.4])
        for name in ("bce", "soft_dice", "dice", "iou", "brier")
    }
    wrong = {
        name: values + 0.1
        for name, values in correct.items()
    }
    wrong.update(
        {
            "probability_mae": np.array([0.05, 0.15]),
            "hard_mask_disagreement": np.array([0.1, 0.2]),
        }
    )

    report = summarize(
        {"correct": correct, "wrong_label": wrong},
        bootstrap_samples=0,
        seed=42,
    )

    assert report["wrong_label"]["vs_correct"]["dice_change"]["mean"] == 0.1
    assert report["wrong_label"]["vs_correct"]["probability_mae"]["mean"] == 0.1


def test_ablation_collation_can_override_stale_embedding_paths() -> None:
    embedding = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    example = {
        "image": Image.new("RGB", (4, 4)),
        "width": 4,
        "height": 4,
        "cache_key": "example",
        "objects": [
            {
                "category": 2,
                "bbox": [0.0, 0.0, 2.0, 2.0],
                "mask": np.zeros((4, 4), dtype=bool),
                "text_embedding_path": "/missing/old-machine/path.pt",
            }
        ],
    }

    batch = collate_online_examples(
        [example],
        image_size=4,
        text_embeddings_by_label={2: embedding},
    )

    torch.testing.assert_close(batch["text_embeddings"], embedding.unsqueeze(0))
