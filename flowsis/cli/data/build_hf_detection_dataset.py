import csv
import shutil
import argparse
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

from flowsis.utils import load_classes


DEFAULT_MANIFEST_PATH = Path("data/manifests/frame_manifest.csv")
DEFAULT_BOXES_DIR = Path("data/bboxes")
DEFAULT_CLASSES_PATH = Path("data/manifests/classes.json")
DEFAULT_OUTPUT_PATH = Path("data/dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Build the Hugging Face detection dataset used by FlowSIS.",
    )
    parser.add_argument("--manifest_path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--boxes_dir", type=Path, default=DEFAULT_BOXES_DIR)
    parser.add_argument("--classes_path", type=Path, default=DEFAULT_CLASSES_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def normalize_image_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def load_boxes(boxes_dir: Path) -> dict[str, NDArray[np.float32]]:
    bboxes: dict[str, NDArray[np.float32]] = {}
    for file in sorted(boxes_dir.glob("*.npy")):
        bboxes[file.stem] = np.load(file).astype(np.float32, copy=False)
    return bboxes


def build_dataset(
    output_path: Path,
    classes: tuple[
        dict[str, str],
        dict[str, int],
        dict[int, str],
    ],
    *,
    manifest_path: Path,
    boxes_dir: Path,
) -> None:
    rows: list[dict[str, object]] = []
    bboxes = load_boxes(boxes_dir)
    vid2label, label2id, id2label = classes

    counters_by_video: dict[str, int] = {}
    missing_bbox_videos: set[str] = set()
    bbox_id = 0

    with manifest_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_id = row["video_id"]
            
            if video_id in bboxes:
                video_boxes = bboxes[video_id]
            else:
                missing_bbox_videos.add(video_id)
                continue

            box_index = counters_by_video.get(video_id, 0)
            if box_index >= len(video_boxes):
                raise ValueError(
                    f"Bounding box count mismatch for video_id={video_id}: "
                    f"requested index {box_index}, but only {len(video_boxes)} boxes are available."
                )

            bbox: NDArray[np.float32] = video_boxes[box_index]
            counters_by_video[video_id] = box_index + 1

            class_name = vid2label[video_id]
            class_id = label2id[class_name]

            rows.append(
                {
                    "image_path": normalize_image_path(row["output_path"]),
                    "image_id": int(row["id"]),
                    "height": int(row["height"]),
                    "width": int(row["width"]),
                    "modified": False,
                    "objects": [
                        {
                            "id": bbox_id,
                            "area": float(bbox[2] * bbox[3]),
                            "bbox": bbox.tolist(),
                            "category": class_id,
                            "video_id": int(video_id),
                            "frame_idx": int(row["frame_idx"]),
                            "modified": False,
                        }
                    ],
                }
            )
            bbox_id += 1

    if not rows:
        raise ValueError("No dataset rows were created. Check the manifest, boxes directory, and class mappings.")

    category_names = [id2label[idx] for idx in range(len(id2label))]
    features = Features(
        {
            "image_path": Value("string"),
            "image_id": Value("int64"),
            "height": Value("int64"),
            "width": Value("int64"),
            "modified": Value("bool"),
            "objects": [
                {
                    "id": Value("int64"),
                    "area": Value("float32"),
                    "bbox": Sequence(Value("float32"), length=4),
                    "category": ClassLabel(names=category_names),
                    "video_id": Value("int64"),
                    "frame_idx": Value("int64"),
                    "modified": Value("bool"),
                }
            ],
        }
    )

    dataset = Dataset.from_list(rows, features=features)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    test_val = split["test"].train_test_split(test_size=0.5, seed=42)
    dataset_splits = DatasetDict(
        {
            "train": split["train"],
            "validation": test_val["train"],
            "test": test_val["test"],
        }
    )

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_splits.save_to_disk(output_path)

    if missing_bbox_videos:
        print(
            "build_dataset_warning",
            {
                "missing_bbox_videos": sorted(missing_bbox_videos),
                "num_missing_videos": len(missing_bbox_videos),
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
        },
    )


def main() -> None:
    args = parse_args()
    classes = load_classes(args.classes_path)
    build_dataset(
        args.output_path,
        classes,
        manifest_path=args.manifest_path,
        boxes_dir=args.boxes_dir,
    )


if __name__ == "__main__":
    main()
