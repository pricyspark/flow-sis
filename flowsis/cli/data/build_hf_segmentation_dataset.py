import csv
import shutil
import argparse
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

from flowsis.data.masks import mask2xywh, load_mask
from flowsis.utils import load_classes


DEFAULT_MANIFEST_PATH = Path("data/manifests/frame_manifest.csv")
DEFAULT_MASKS_DIR = Path("data/masks")
DEFAULT_CLASSES_PATH = Path("data/manifests/classes.json")
DEFAULT_TEXT_EMBEDDINGS_DIR = Path("data/manifests/text-embeddings")
DEFAULT_FEATURE_CACHE_DIR = Path("data/cache/base")
DEFAULT_OUTPUT_PATH = Path("data/segmentation-dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Hugging Face segmentation dataset used by FlowSIS.",
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR)
    parser.add_argument("--classes-path", type=Path, default=DEFAULT_CLASSES_PATH)
    parser.add_argument("--text-embeddings-dir", type=Path, default=DEFAULT_TEXT_EMBEDDINGS_DIR)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=DEFAULT_FEATURE_CACHE_DIR,
        help=(
            "Root directory for cached detector feature maps. The builder stores a "
            "per-example cache_dir path for a future versioned feature_bundle.pt, "
            "but does not require the directory to exist yet."
        ),
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    return parser.parse_args()


def normalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def build_cache_key(video_id: str, frame_idx: str, image_id: str) -> str:
    return f"video_{int(video_id):06d}_frame_{int(frame_idx):06d}_image_{int(image_id):08d}"


def validate_split_sizes(train_size: float, validation_size: float, test_size: float) -> None:
    total = train_size + validation_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "Split sizes must sum to 1.0, "
            f"but received train={train_size}, validation={validation_size}, test={test_size}."
        )


def build_dataset(
    output_path: Path,
    classes: tuple[
        dict[str, str],
        dict[str, int],
        dict[int, str],
    ],
    *,
    manifest_path: Path,
    masks_dir: Path,
    text_embeddings_dir: Path,
    feature_cache_dir: Path,
    train_size: float,
    validation_size: float,
    test_size: float,
) -> None:
    validate_split_sizes(train_size, validation_size, test_size)

    rows: list[dict[str, object]] = []
    missing_mask_examples: list[dict[str, int]] = []
    missing_text_labels: set[str] = set()
    vid2label, label2id, id2label = classes

    with manifest_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_id = row["video_id"]
            frame_idx = row["frame_idx"]
            image_id = row["id"]

            if video_id not in vid2label:
                continue

            mask_path = masks_dir / video_id / f"{frame_idx}.npz"
            if not mask_path.exists():
                missing_mask_examples.append(
                    {
                        "video_id": int(video_id),
                        "frame_idx": int(frame_idx),
                        "image_id": int(image_id),
                    }
                )
                continue

            label = vid2label[video_id]
            category_id = label2id[label]
            text_embedding_path = text_embeddings_dir / f"{label}.pt"
            if not text_embedding_path.exists():
                missing_text_labels.add(label)
                continue

            mask = load_mask(int(video_id), int(frame_idx), masks_dir)
            bbox = mask2xywh(mask)
            if bbox is None:
                continue

            cache_key = build_cache_key(video_id, frame_idx, image_id)
            cache_dir = feature_cache_dir / cache_key
            image_path = normalize_path(row["output_path"])

            rows.append(
                {
                    "image_path": image_path,
                    "image_id": int(image_id),
                    "height": int(row["height"]),
                    "width": int(row["width"]),
                    "modified": False,
                    "cache_key": cache_key,
                    "cache_dir": normalize_path(cache_dir),
                    "objects": [
                        {
                            "id": int(image_id),
                            "area": float(bbox[2] * bbox[3]),
                            "bbox": [float(value) for value in bbox],
                            "category": category_id,
                            "video_id": int(video_id),
                            "frame_idx": int(frame_idx),
                            "mask_path": normalize_path(mask_path),
                            "text_embedding_path": normalize_path(text_embedding_path),
                            "modified": False,
                        }
                    ],
                }
            )

    if not rows:
        raise ValueError(
            "No segmentation dataset rows were created. Check the manifest, masks, classes, and text embeddings."
        )

    category_names = [id2label[idx] for idx in range(len(id2label))]
    features = Features(
        {
            "image_path": Value("string"),
            "image_id": Value("int64"),
            "height": Value("int64"),
            "width": Value("int64"),
            "modified": Value("bool"),
            "cache_key": Value("string"),
            "cache_dir": Value("string"),
            "objects": [
                {
                    "id": Value("int64"),
                    "area": Value("float32"),
                    "bbox": Sequence(Value("float32"), length=4),
                    "category": ClassLabel(names=category_names),
                    "video_id": Value("int64"),
                    "frame_idx": Value("int64"),
                    "mask_path": Value("string"),
                    "text_embedding_path": Value("string"),
                    "modified": Value("bool"),
                }
            ],
        }
    )

    dataset = Dataset.from_list(rows, features=features)
    held_out_size = validation_size + test_size
    split = dataset.train_test_split(test_size=held_out_size, seed=42)
    validation_ratio = validation_size / held_out_size if held_out_size > 0 else 0.5
    held_out = split["test"].train_test_split(test_size=1.0 - validation_ratio, seed=42)
    dataset_splits = DatasetDict(
        {
            "train": split["train"],
            "validation": held_out["train"],
            "test": held_out["test"],
        }
    )

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_splits.save_to_disk(output_path)

    if missing_mask_examples:
        print(
            "build_dataset_warning",
            {
                "missing_mask_examples": missing_mask_examples[:10],
                "num_missing_mask_examples": len(missing_mask_examples),
            },
        )
    if missing_text_labels:
        print(
            "build_dataset_warning",
            {
                "missing_text_labels": sorted(missing_text_labels),
                "num_missing_text_labels": len(missing_text_labels),
            },
        )

    print(
        "build_dataset",
        {
            "output_path": str(output_path),
            "num_rows": len(dataset),
            "num_train": len(dataset_splits["train"]),
            "num_validation": len(dataset_splits["validation"]),
            "num_test": len(dataset_splits["test"]),
            "feature_cache_root": str(feature_cache_dir),
            "notes": [
                "Segmentation labels, masks, and prompt embeddings are object-level fields.",
                "Online augmentation may add objects; every surviving object becomes a mask query.",
                "Offline training expects versioned feature_bundle.pt artifacts; "
                "cache generation is not implemented yet.",
                "For staged training, reuse the same dataset and switch only the training phase.",
            ],
        },
    )


def main() -> None:
    args = parse_args()
    classes = load_classes(args.classes_path)
    build_dataset(
        args.output_path,
        classes,
        manifest_path=args.manifest_path,
        masks_dir=args.masks_dir,
        text_embeddings_dir=args.text_embeddings_dir,
        feature_cache_dir=args.feature_cache_dir,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
