"""Measure how much a trained base head depends on its text prompts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from datasets import ClassLabel, Dataset, DatasetDict
from torch.utils.data import DataLoader

from flowsis.artifacts import atomic_write_text
from flowsis.base_head import BaseFusionHead
from flowsis.cli.common import (
    add_detector_arguments,
    dataset_from_args,
    ensure_split_exists,
)
from flowsis.cli.train.train_base_head import (
    collate_online_examples,
    extract_online_detector_output,
    load_segmentation_objects,
    load_text_embedding,
    match_object_queries,
)
from flowsis.data import CallablePipeline, PreparedDataset, load_object_image
from flowsis.data.augment import center_square_augment
from flowsis.data.object_records import get_object_feature_schema, get_object_records
from flowsis.head_checkpoint import load_head
from flowsis.pretrained import Detector, load_detector
from flowsis.utils import build_autocast_context, get_device, set_seed

ABLATIONS = ("correct", "wrong_label", "scrambled", "zero")
QUALITY_METRICS = ("bce", "soft_dice", "dice", "iou", "brier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="data/segmentation-dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--head-path", type=Path, default=Path("outputs/base/final"))
    parser.add_argument(
        "--text-embeddings-dir",
        type=Path,
        default=Path("data/manifests/text-embeddings"),
    )
    parser.add_argument("--images-dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/masks"))
    add_detector_arguments(
        parser,
        model_flag="--detector-model",
        model_dest="detector_model_source",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=ABLATIONS[1:],
        default=list(ABLATIONS[1:]),
        help=(
            "Controls to compare with the correct prompt. The correct condition "
            "is always run."
        ),
    )
    parser.add_argument(
        "--output-path", type=Path, default=Path("outputs/text-ablation.json")
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Evaluate only the first N images, for a quick smoke test.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def label_names(split: Dataset) -> dict[int, str]:
    schema = get_object_feature_schema(split.features["objects"])
    category = getattr(schema["category"], "feature", schema["category"])
    if not isinstance(category, ClassLabel):
        raise TypeError("Expected objects.category to be a ClassLabel feature.")
    return {
        label: name or f"class_{label}"
        for label, name in enumerate(category.names)
    }


def load_label_embeddings(
    dataset: DatasetDict,
    directory: Path,
) -> dict[int, torch.Tensor]:
    """Load one canonical prompt tensor for each class in the dataset."""
    first_split = next(iter(dataset.values()), None)
    if first_split is None:
        raise ValueError("The dataset contains no splits.")
    names = label_names(first_split)
    return {
        label: load_text_embedding(
            {"text_embedding_path": str(directory / f"{name}.pt")}
        )
        for label, name in names.items()
    }


def relocate_example_paths(
    example: dict[str, Any],
    *,
    images_dir: Path,
    masks_dir: Path,
    **_: Any,
) -> dict[str, Any]:
    """Resolve relocatable image and mask paths from object metadata."""
    objects = get_object_records(example)
    if not objects:
        return example
    image_video_id = int(objects[0]["video_id"])
    example["image_path"] = str(
        images_dir / str(image_video_id) / Path(example["image_path"]).name
    )
    for obj in objects:
        obj["mask_path"] = str(
            masks_dir / str(int(obj["video_id"])) / Path(obj["mask_path"]).name
        )
    example["objects"] = objects
    return example


def build_evaluation_dataset(
    split: Dataset,
    *,
    image_size: int,
    images_dir: Path,
    masks_dir: Path,
) -> Dataset:
    return cast(
        Dataset,
        PreparedDataset(
            split,
            loader=CallablePipeline(
                (relocate_example_paths, load_object_image, load_segmentation_objects),
                ({"images_dir": images_dir, "masks_dir": masks_dir}, {}, {}),
            ),
            augment=CallablePipeline(
                (center_square_augment,), ({"crop_size": image_size},)
            ),
        ),
    )


def make_wrong_label_map(labels: list[int], *, seed: int) -> dict[int, int]:
    """Return a seeded derangement so no class keeps its own prompt."""
    ordered = sorted(set(labels))
    if len(ordered) < 2:
        raise ValueError("The wrong-label ablation requires at least two classes.")
    generator = torch.Generator().manual_seed(seed)
    for _ in range(1000):
        shuffled = torch.randperm(len(ordered), generator=generator).tolist()
        if all(index != shuffled[index] for index in range(len(ordered))):
            return {
                label: ordered[shuffled[index]]
                for index, label in enumerate(ordered)
            }
    # This is only reachable through spectacularly unlikely bad luck.
    return {
        label: ordered[(index + 1) % len(ordered)]
        for index, label in enumerate(ordered)
    }


def ablate_embeddings(
    condition: str,
    correct: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_embeddings: Mapping[int, torch.Tensor],
    wrong_label_map: Mapping[int, int],
    feature_permutation: torch.Tensor,
) -> torch.Tensor:
    if condition == "correct":
        return correct
    if condition == "zero":
        return torch.zeros_like(correct)
    if condition == "scrambled":
        return correct.index_select(-1, feature_permutation.to(correct.device))
    if condition != "wrong_label":
        raise ValueError(f"Unknown text ablation {condition!r}.")

    replacements = []
    for label in labels.detach().cpu().tolist():
        replacement_label = wrong_label_map[int(label)]
        replacement = label_embeddings[replacement_label]
        if replacement.shape != correct.shape[1:]:
            raise ValueError(
                "Wrong-label prompt tensors must have the same shape as the correct "
                f"batch prompts. Expected {tuple(correct.shape[1:])}, got "
                f"{tuple(replacement.shape)} for class {replacement_label}."
            )
        replacements.append(replacement)
    return torch.stack(replacements).to(device=correct.device, dtype=correct.dtype)


def per_object_metrics(
    mask_logits: torch.Tensor,
    target_masks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    reduce_dims = tuple(range(1, mask_logits.ndim))
    probabilities = mask_logits.float().sigmoid()
    targets = target_masks.float()
    bce = F.binary_cross_entropy_with_logits(
        mask_logits.float(), targets, reduction="none"
    ).mean(dim=reduce_dims)
    intersection = (probabilities * targets).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
    soft_dice = (2.0 * intersection + 1.0) / (denominator + 1.0)

    prediction = probabilities >= 0.5
    target = targets >= 0.5
    hard_intersection = (prediction & target).sum(dim=reduce_dims).float()
    prediction_area = prediction.sum(dim=reduce_dims).float()
    target_area = target.sum(dim=reduce_dims).float()
    union = (prediction | target).sum(dim=reduce_dims).float()
    return {
        "bce": bce,
        "soft_dice": soft_dice,
        "dice": (2.0 * hard_intersection + 1.0)
        / (prediction_area + target_area + 1.0),
        "iou": (hard_intersection + 1.0) / (union + 1.0),
        "brier": ((probabilities - targets) ** 2).mean(dim=reduce_dims),
    }


def mean_with_ci(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    generator: np.random.Generator,
) -> dict[str, float | list[float]]:
    mean = float(values.mean())
    if bootstrap_samples == 0 or len(values) == 1:
        interval = [mean, mean]
    else:
        bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
        for start in range(0, bootstrap_samples, 256):
            count = min(256, bootstrap_samples - start)
            indices = generator.integers(0, len(values), size=(count, len(values)))
            bootstrap_means[start : start + count] = values[indices].mean(axis=1)
        interval = np.quantile(bootstrap_means, (0.025, 0.975)).tolist()
    return {"mean": mean, "ci95": interval}


def prepare_head_inputs(
    head: BaseFusionHead,
    detector: Detector,
    batch: dict[str, Any],
) -> tuple[list[torch.Tensor], torch.Tensor | None, torch.Tensor, torch.Tensor]:
    device = next(head.parameters()).device
    detector_output = extract_online_detector_output(detector, batch["images"])
    object_image_indices = batch["object_image_indices"].to(device, non_blocking=True)
    object_labels = batch["object_labels"].to(device, non_blocking=True)
    object_boxes = batch["object_boxes"].to(device, non_blocking=True).clamp(0.0, 1.0)
    object_queries = None

    if head.query_dim is not None:
        query_embeddings = detector_output.query_embeddings
        query_logits = detector_output.query_logits
        query_boxes = detector_output.query_boxes
        if not all(
            isinstance(value, torch.Tensor)
            for value in (query_embeddings, query_logits, query_boxes)
        ):
            raise RuntimeError(
                "The detector does not expose the queries, logits, and boxes required "
                "by this head checkpoint."
            )
        typed_embeddings = cast(torch.Tensor, query_embeddings)
        if typed_embeddings.shape[-1] != head.query_dim:
            raise RuntimeError(
                f"Detector query dimension {typed_embeddings.shape[-1]} does not "
                f"match head query dimension {head.query_dim}."
            )
        matched = match_object_queries(
            cast(torch.Tensor, query_logits),
            cast(torch.Tensor, query_boxes),
            object_image_indices,
            object_labels,
            object_boxes,
        )
        object_queries = typed_embeddings[object_image_indices, matched]

    features = [
        feature.to(device, non_blocking=True).index_select(0, object_image_indices)
        for feature in detector_output.feature_maps
    ]
    return features, object_queries, object_boxes, object_labels


@torch.inference_mode()
def evaluate(
    head: BaseFusionHead,
    detector: Detector,
    data_loader: DataLoader,
    conditions: list[str],
    *,
    label_embeddings: Mapping[int, torch.Tensor],
    wrong_label_map: Mapping[int, int],
    feature_permutation: torch.Tensor,
    use_amp: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    head.eval()
    detector.eval()
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        condition: {metric: [] for metric in QUALITY_METRICS}
        for condition in conditions
    }
    for condition in conditions[1:]:
        collected[condition].update(
            {"probability_mae": [], "hard_mask_disagreement": []}
        )

    num_images = 0
    num_objects = 0
    for batch_index, batch in enumerate(data_loader, start=1):
        features, object_queries, object_boxes, object_labels = prepare_head_inputs(
            head, detector, batch
        )
        device = next(head.parameters()).device
        correct_embeddings = batch["text_embeddings"].to(device, non_blocking=True)
        target_masks = batch["target_masks"].to(device, non_blocking=True)
        correct_probabilities = None

        for condition in conditions:
            text_embeddings = ablate_embeddings(
                condition,
                correct_embeddings,
                object_labels,
                label_embeddings=label_embeddings,
                wrong_label_map=wrong_label_map,
                feature_permutation=feature_permutation,
            )
            with build_autocast_context(enabled=use_amp, device=device):
                output = head(
                    features,
                    text_embeddings,
                    object_boxes=object_boxes,
                    object_queries=object_queries,
                    mask_output_size=target_masks.shape[-2:],
                    return_intermediates=False,
                )
            mask_logits = cast(torch.Tensor, output["mask_logits"])
            metrics = per_object_metrics(mask_logits, target_masks)
            for name, values in metrics.items():
                collected[condition][name].append(values.cpu().numpy())

            probabilities = mask_logits.float().sigmoid()
            if condition == "correct":
                correct_probabilities = probabilities
                continue
            if correct_probabilities is None:
                raise RuntimeError("The correct condition must run before ablations.")
            reduce_dims = tuple(range(1, probabilities.ndim))
            collected[condition]["probability_mae"].append(
                (probabilities - correct_probabilities)
                .abs()
                .mean(dim=reduce_dims)
                .cpu()
                .numpy()
            )
            collected[condition]["hard_mask_disagreement"].append(
                ((probabilities >= 0.5) != (correct_probabilities >= 0.5))
                .float()
                .mean(dim=reduce_dims)
                .cpu()
                .numpy()
            )

        num_images += len(batch["cache_keys"])
        num_objects += len(object_labels)
        print(
            "text_ablation_progress",
            json.dumps(
                {
                    "batches": batch_index,
                    "images": num_images,
                    "objects": num_objects,
                }
            ),
            flush=True,
        )

    arrays = {
        condition: {
            name: np.concatenate(chunks)
            for name, chunks in condition_metrics.items()
        }
        for condition, condition_metrics in collected.items()
    }
    return arrays, num_images


def summarize(
    values: dict[str, dict[str, np.ndarray]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    correct = values["correct"]
    summary: dict[str, Any] = {}
    for condition, metrics in values.items():
        condition_summary: dict[str, Any] = {
            "metrics": {
                name: float(metric_values.mean())
                for name, metric_values in metrics.items()
                if name in QUALITY_METRICS
            }
        }
        if condition != "correct":
            condition_summary["vs_correct"] = {
                f"{name}_change": mean_with_ci(
                    metrics[name] - correct[name],
                    bootstrap_samples=bootstrap_samples,
                    generator=generator,
                )
                for name in QUALITY_METRICS
            }
            condition_summary["vs_correct"].update(
                {
                    name: mean_with_ci(
                        metrics[name],
                        bootstrap_samples=bootstrap_samples,
                        generator=generator,
                    )
                    for name in ("probability_mae", "hard_mask_disagreement")
                }
            )
        summary[condition] = condition_summary
    return summary


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError(
            "--batch-size must be positive and --num-workers non-negative."
        )
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive when provided.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative.")
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else get_device()
    use_amp = bool(args.amp and device.type == "cuda")
    dataset = dataset_from_args(args)
    ensure_split_exists(dataset, args.split, role="evaluation")
    split = dataset[args.split]
    if args.max_images is not None:
        split = split.select(range(min(args.max_images, len(split))))
    if not len(split):
        raise ValueError(f"Evaluation split {args.split!r} is empty.")

    head, checkpoint_path = load_head(args.head_path, device=device)
    detector = load_detector(
        args.detector_model_source,
        architecture=args.detector_architecture,
        image_size=args.image_size,
        device=device,
    )
    detector.eval()
    detector.requires_grad_(False)
    head.eval()

    label_embeddings = load_label_embeddings(dataset, args.text_embeddings_dir)
    conditions = ["correct", *dict.fromkeys(args.ablations)]
    wrong_label_map: dict[int, int] = {}
    if "wrong_label" in conditions:
        wrong_label_map = make_wrong_label_map(list(label_embeddings), seed=args.seed)

    first_embedding = next(iter(label_embeddings.values()), None)
    if first_embedding is None:
        raise ValueError("The dataset contains no object prompt embeddings.")
    feature_permutation = torch.randperm(
        first_embedding.shape[-1], generator=torch.Generator().manual_seed(args.seed)
    )
    prepared_split = build_evaluation_dataset(
        split,
        image_size=args.image_size,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
    )
    generator = torch.Generator().manual_seed(args.seed)
    data_loader = DataLoader(
        prepared_split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(
            collate_online_examples,
            image_size=args.image_size,
            text_embeddings_by_label=label_embeddings,
        ),
        generator=generator,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    values, num_images = evaluate(
        head,
        detector,
        data_loader,
        conditions,
        label_embeddings=label_embeddings,
        wrong_label_map=wrong_label_map,
        feature_permutation=feature_permutation,
        use_amp=use_amp,
    )
    names = label_names(dataset[args.split])
    report = {
        "experiment": "text_conditioning_ablation",
        "dataset_path": str(args.dataset_path),
        "split": args.split,
        "num_images": num_images,
        "num_objects": len(values["correct"]["dice"]),
        "head_checkpoint": str(checkpoint_path),
        "text_embeddings_dir": str(args.text_embeddings_dir),
        "images_dir": str(args.images_dir),
        "masks_dir": str(args.masks_dir),
        "detector_architecture": detector.architecture,
        "detector_model": detector.source,
        "device": str(device),
        "amp": use_amp,
        "image_size": args.image_size,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "conditions": conditions,
        "wrong_label_map": {
            names.get(source, str(source)): names.get(target, str(target))
            for source, target in wrong_label_map.items()
        },
        "interpretation": {
            "metric_change": (
                "ablation minus correct; negative Dice/IoU changes mean worse masks"
            ),
            "probability_mae": (
                "mean absolute pixel-probability change from the correct prompt"
            ),
            "hard_mask_disagreement": (
                "fraction of pixels whose thresholded prediction changed"
            ),
            "isolation": (
                "detector features, boxes, and matched object queries are held fixed "
                "across prompts"
            ),
        },
        "results": summarize(
            values,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    atomic_write_text(args.output_path, json.dumps(report, indent=2) + "\n")
    print("text_ablation_summary", json.dumps(report, sort_keys=True), flush=True)
    print(f"Saved report to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
