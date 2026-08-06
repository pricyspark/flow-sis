"""Evaluate a supported detector checkpoint with score-ranked PR curves."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict, load_from_disk

from flowsis.data.images import get_image
from flowsis.data.object_records import get_object_feature_schema, get_object_records
from flowsis.cli.common import add_detector_arguments
from flowsis.pretrained import load_detector
from flowsis.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_detector_arguments(parser)
    parser.add_argument("--dataset-path", default="data/dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="outputs/detector-evaluation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty(0, dtype=np.float64)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=1)
    box_area = np.prod(np.maximum(box[2:] - box[:2], 0.0))
    areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0.0), axis=1)
    return intersection / np.maximum(
        box_area + areas - intersection, np.finfo(float).eps
    )


def precision_recall(
    records: list[tuple[float, bool]], positives: int
) -> dict[str, Any]:
    records.sort(key=lambda record: record[0], reverse=True)
    scores = np.asarray([record[0] for record in records], dtype=np.float64)
    true_positive = np.asarray([record[1] for record in records], dtype=np.float64)
    tp = np.cumsum(true_positive)
    fp = np.cumsum(1.0 - true_positive)
    recall = tp / positives if positives else np.zeros_like(tp)
    precision = tp / np.maximum(tp + fp, np.finfo(float).eps)

    # COCO/VOC-style interpolated area under the precision-recall envelope.
    padded_recall = np.concatenate(([0.0], recall, [1.0]))
    padded_precision = np.concatenate(([1.0], precision, [0.0]))
    padded_precision = np.maximum.accumulate(padded_precision[::-1])[::-1]
    changes = np.flatnonzero(padded_recall[1:] != padded_recall[:-1])
    ap = float(
        np.sum(
            (padded_recall[changes + 1] - padded_recall[changes])
            * padded_precision[changes + 1]
        )
    )
    return {"scores": scores, "precision": precision, "recall": recall, "ap": ap}


def label_names(split: Dataset) -> dict[int, str]:
    schema = get_object_feature_schema(split.features["objects"])
    category = schema["category"]
    category = getattr(category, "feature", category)
    if not isinstance(category, ClassLabel):
        raise TypeError("Expected objects.category to be a ClassLabel.")
    return {
        index: name or f"class_{index}" for index, name in enumerate(category.names)
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be between 0 and 1.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_from_disk(args.dataset_path)
    if not isinstance(loaded, DatasetDict):
        raise TypeError("Expected --dataset-path to contain a DatasetDict.")
    if args.split not in loaded:
        raise KeyError(f"Split {args.split!r} is not present in the dataset.")
    split = loaded[args.split]
    names = label_names(split)
    device = torch.device(args.device) if args.device else get_device()
    model = load_detector(
        args.model_source,
        architecture=args.detector_architecture,
        image_size=args.image_size,
        device=device,
    )
    architecture = model.architecture

    ground_truth: dict[int, dict[int, np.ndarray]] = {}
    positives: defaultdict[int, int] = defaultdict(int)
    predictions: list[tuple[float, int, int, np.ndarray]] = []

    for start in range(0, len(split), args.batch_size):
        examples = [
            split[index]
            for index in range(
                start,
                min(start + args.batch_size, len(split)),
            )
        ]
        images = [get_image(example, convert_mode="RGB") for example in examples]
        result = model.infer(
            images,
            threshold=args.score_threshold,
            device_preprocess=True,
        )
        for offset, (example, detection) in enumerate(zip(examples, result.detections)):
            image_index = start + offset
            boxes_by_class: defaultdict[int, list[list[float]]] = defaultdict(list)
            for obj in get_object_records(example):
                x, y, width, height = (float(value) for value in obj["bbox"])
                label = int(obj["category"])
                boxes_by_class[label].append([x, y, x + width, y + height])
                positives[label] += 1
            ground_truth[image_index] = {
                label: np.asarray(boxes, dtype=np.float64)
                for label, boxes in boxes_by_class.items()
            }
            for score, label, box in zip(
                detection["scores"].detach().cpu().tolist(),
                detection["labels"].detach().cpu().tolist(),
                detection["boxes"].detach().cpu().tolist(),
            ):
                predictions.append(
                    (
                        float(score),
                        int(label),
                        image_index,
                        np.asarray(box, dtype=np.float64),
                    )
                )
        print(
            f"evaluated_images {min(start + args.batch_size, len(split))}/{len(split)}",
            flush=True,
        )

    class_records: defaultdict[int, list[tuple[float, bool]]] = defaultdict(list)
    matched: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for score, label, image_index, box in sorted(
        predictions,
        reverse=True,
        key=lambda item: item[0],
    ):
        candidates = ground_truth[image_index].get(
            label, np.empty((0, 4), dtype=np.float64)
        )
        ious = box_iou(box, candidates)
        is_true_positive = False
        if ious.size:
            candidate_index = int(np.argmax(ious))
            key = (image_index, label)
            if (
                ious[candidate_index] >= args.iou_threshold
                and candidate_index not in matched[key]
            ):
                matched[key].add(candidate_index)
                is_true_positive = True
        class_records[label].append((score, is_true_positive))

    metrics: dict[int, dict[str, Any]] = {}
    for label in sorted(names):
        metrics[label] = precision_recall(class_records[label], positives[label])
    micro_records = [
        record for label in sorted(names) for record in class_records[label]
    ]
    micro = precision_recall(micro_records, sum(positives.values()))

    with (output_dir / "pr_curve.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "score", "recall", "precision"])
        for label, metric in metrics.items():
            for score, recall, precision in zip(
                metric["scores"],
                metric["recall"],
                metric["precision"],
            ):
                writer.writerow([label, names[label], score, recall, precision])

    summary = {
        "architecture": architecture,
        "model": model.source,
        "dataset_path": str(args.dataset_path),
        "split": args.split,
        "num_images": len(split),
        "iou_threshold": args.iou_threshold,
        "score_threshold": args.score_threshold,
        "num_ground_truth": sum(positives.values()),
        "num_predictions": len(predictions),
        "micro_ap": micro["ap"],
        "macro_ap": float(np.mean([metric["ap"] for metric in metrics.values()])),
        "classes": {
            names[label]: {
                "class_id": label,
                "ground_truth": positives[label],
                "predictions": len(class_records[label]),
                "ap": metric["ap"],
            }
            for label, metric in metrics.items()
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axis = plt.subplots(figsize=(8, 7))
    for label, metric in metrics.items():
        axis.plot(
            metric["recall"],
            metric["precision"],
            label=f"{names[label]} (AP={metric['ap']:.3f})",
        )
    axis.plot(
        micro["recall"],
        micro["precision"],
        "k--",
        linewidth=2,
        label=f"micro (AP={micro['ap']:.3f})",
    )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.01),
        xlabel="Recall",
        ylabel="Precision",
        title=f"{architecture} precision-recall at IoU {args.iou_threshold:.2f}",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left", fontsize="small")
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall.png", dpi=180)
    plt.close(figure)
    print("evaluation_summary", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
