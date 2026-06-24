import argparse
import math
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datasets import ClassLabel, Dataset, DatasetDict, load_from_disk

from flowsis.data.images import get_image
from flowsis.data.object_records import get_object_feature_schema, get_object_records


DEFAULT_DATASET_PATH = Path("data/dataset")
DEFAULT_OUTPUT_DIR = Path("outputs/dataset_visualization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize samples from the saved HF object detection dataset."
    )
    parser.add_argument("--dataset_path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle samples before selecting them.",
    )
    parser.add_argument(
        "--cell_size",
        type=int,
        default=480,
        help="Longest edge used for each tile in the contact sheet.",
    )
    return parser.parse_args()


def load_split(dataset_path: Path, split_name: str) -> tuple[Dataset, dict[int, str]]:
    dataset = load_from_disk(str(dataset_path))
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected a dataset with named splits.")
    if split_name not in dataset:
        raise KeyError(f"Split '{split_name}' not found in dataset. Available: {list(dataset.keys())}")

    split = dataset[split_name]
    object_schema = get_object_feature_schema(split.features["objects"])
    category_feature = object_schema["category"]
    if hasattr(category_feature, "feature"):
        category_feature = category_feature.feature
    if not isinstance(category_feature, ClassLabel):
        raise TypeError("Expected objects.category to be a ClassLabel feature.")

    id2label = {index: name for index, name in enumerate(category_feature.names)}
    return split, id2label


def select_indices(length: int, *, num_samples: int, shuffle: bool, seed: int) -> list[int]:
    indices = list(range(length))
    if shuffle:
        random.Random(seed).shuffle(indices)
    return indices[: min(num_samples, length)]


def draw_example(example: dict, *, id2label: dict[int, str]) -> Image.Image:
    image = get_image(example, convert_mode="RGB")
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    object_records = get_object_records(example)
    colors = [
        "#ff5a5f",
        "#2ec4b6",
        "#ffbf69",
        "#3867d6",
        "#6a4c93",
        "#20bf55",
    ]

    for index, object_record in enumerate(object_records):
        x, y, width, height = [float(value) for value in object_record["bbox"]]
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(canvas.width), x + width)
        y2 = min(float(canvas.height), y + height)
        color = colors[index % len(colors)]

        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, canvas.width // 300))

        category_id = int(object_record["category"])
        label = id2label.get(category_id, str(category_id))
        text = f"{label}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top
        text_x = x1
        text_y = max(0.0, y1 - text_height - 6)
        draw.rectangle(
            (text_x, text_y, text_x + text_width + 8, text_y + text_height + 4),
            fill=color,
        )
        draw.text((text_x + 4, text_y + 2), text, fill="white", font=font)

    source_pairs = {
        (int(record["video_id"]), int(record["frame_idx"]))
        for record in object_records
        if "video_id" in record and "frame_idx" in record
    }
    if len(source_pairs) == 1:
        video_id, frame_idx = next(iter(source_pairs))
        source_summary = f"video_id={video_id} frame_idx={frame_idx}"
    else:
        source_summary = f"sources={len(source_pairs)}"

    header = (
        f"image_id={example['image_id']} "
        f"{source_summary} "
        f"objects={len(object_records)}"
    )
    left, top, right, bottom = draw.textbbox((0, 0), header, font=font)
    header_width = right - left
    header_height = bottom - top
    draw.rectangle((0, 0, header_width + 12, header_height + 8), fill="black")
    draw.text((6, 4), header, fill="white", font=font)

    return canvas


def make_contact_sheet(images: list[Image.Image], *, cell_size: int) -> Image.Image:
    if not images:
        raise ValueError("Expected at least one image for the contact sheet.")

    columns = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * cell_size, rows * cell_size), color=(24, 24, 24))

    for index, image in enumerate(images):
        tile = image.copy()
        tile.thumbnail((cell_size, cell_size))
        offset_x = (index % columns) * cell_size + (cell_size - tile.width) // 2
        offset_y = (index // columns) * cell_size + (cell_size - tile.height) // 2
        sheet.paste(tile, (offset_x, offset_y))

    return sheet


def main() -> None:
    args = parse_args()
    split, id2label = load_split(args.dataset_path, args.split)
    indices = select_indices(len(split), num_samples=args.num_samples, shuffle=args.shuffle, seed=args.seed)
    if not indices:
        raise ValueError(
            f"No samples selected from split '{args.split}'. "
            f"Dataset length={len(split)}, requested num_samples={args.num_samples}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rendered_images: list[Image.Image] = []
    for dataset_index in indices:
        example = split[dataset_index]
        rendered = draw_example(example, id2label=id2label)
        rendered_images.append(rendered)

        output_path = args.output_dir / f"{args.split}_{dataset_index:05d}_image_{int(example['image_id']):06d}.jpg"
        rendered.save(output_path, quality=95)

    contact_sheet = make_contact_sheet(rendered_images, cell_size=args.cell_size)
    contact_sheet_path = args.output_dir / f"{args.split}_contact_sheet.jpg"
    contact_sheet.save(contact_sheet_path, quality=95)

    print(
        "visualized_dataset",
        {
            "dataset_path": str(args.dataset_path),
            "split": args.split,
            "num_samples": len(indices),
            "output_dir": str(args.output_dir),
            "contact_sheet": str(contact_sheet_path),
        },
    )


if __name__ == "__main__":
    main()
