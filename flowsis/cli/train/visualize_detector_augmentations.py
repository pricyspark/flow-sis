import argparse
import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from PIL import Image, ImageDraw, ImageFont

from flowsis.data import PreparedDataset
from flowsis.data.images import get_image
from flowsis.data.object_records import get_object_records
from flowsis.data.augment import AugmentationStep
from flowsis.cli.common import dataset_from_args
from flowsis.cli.train.train_detector import (
    build_detection_loader,
    build_train_augmentation_steps,
    build_validation_augmentation_steps,
    load_label_metadata,
)


DEFAULT_OUTPUT_DIR = Path("outputs/detector-augmentation-visualization")
MASK_ALPHA = 88
COLORS = (
    "#ff5a5f",
    "#2ec4b6",
    "#ffbf69",
    "#3867d6",
    "#6a4c93",
    "#20bf55",
    "#ff7f50",
    "#17a2b8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the RT-DETRv2 online augmentation pipeline with raw, "
            "loaded, intermediate, and final stages."
        )
    )
    parser.add_argument("--dataset-path", type=str, default="data/dataset")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--validation-split", type=str, default="validation")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pipeline",
        choices=("train", "validation", "both"),
        default="both",
        help="Which augmentation pipeline to visualize.",
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle indices before selecting samples when --indices is not provided.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=480,
        help="Longest edge used for each stage tile in the saved strips.",
    )
    parser.add_argument(
        "--save-stage-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save each intermediate stage image in addition to the combined strip.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass a per-sample RNG through the pipeline so repeated runs are reproducible.",
    )
    parser.add_argument(
        "--use-rotation-augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply mask-guided rotation augmentation during training visualization.",
    )
    parser.add_argument(
        "--use-roi-square-augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply object-centered square cropping during training visualization.",
    )
    parser.add_argument(
        "--use-overlap-augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply overlap compositing during training visualization.",
    )
    parser.add_argument(
        "--overlap-min-overlays",
        type=int,
        default=1,
        help="Minimum number of samples to composite when overlap augmentation is enabled.",
    )
    parser.add_argument(
        "--overlap-max-overlays",
        type=int,
        default=1,
        help="Maximum number of samples to composite when overlap augmentation is enabled.",
    )
    parser.add_argument(
        "--overlap-p",
        type=float,
        default=0.5,
        help="Geometric continuation parameter used to sample additional overlap layers.",
    )
    parser.add_argument(
        "--use-photometric-augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply photometric augmentation during training visualization.",
    )
    return parser.parse_args()


def select_indices(
    split_dataset: Dataset,
    *,
    indices: list[int] | None,
    num_samples: int,
    shuffle: bool,
    seed: int,
) -> list[int]:
    if indices is not None and len(indices) > 0:
        max_index = len(split_dataset) - 1
        for index in indices:
            if index < 0 or index > max_index:
                raise IndexError(f"Requested index {index} is out of bounds for dataset length {len(split_dataset)}.")
        return indices

    candidates = list(range(len(split_dataset)))
    if shuffle:
        random.Random(seed).shuffle(candidates)
    return candidates[: min(num_samples, len(candidates))]


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def clone_example(example: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(example)


def capture_pipeline_stages(
    prepared_dataset: PreparedDataset,
    idx: int,
    *,
    steps: list[AugmentationStep],
    deterministic: bool,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    raw_example = prepared_dataset.get_raw_example(idx)
    loaded_example = prepared_dataset.load_example(clone_example(raw_example))
    current_example = clone_example(loaded_example)
    context = prepared_dataset.build_context(idx)

    snapshots: list[tuple[str, dict[str, Any]]] = [
        ("raw", clone_example(raw_example)),
        ("loaded", clone_example(loaded_example)),
    ]

    for step_index, (name, callable_, kwargs) in enumerate(steps, start=1):
        runtime_kwargs = {"context": context, **kwargs}
        if deterministic:
            runtime_kwargs["seed"] = seed + idx * 1000 + step_index
        result = callable_(current_example, **runtime_kwargs)
        if result is not None:
            current_example = result

        suffix = " (final)" if step_index == len(steps) else ""
        snapshots.append((f"{step_index}. {name}{suffix}", clone_example(current_example)))

    if not steps:
        snapshots.append(("final", clone_example(current_example)))

    return snapshots


def render_stage(
    example: dict[str, Any],
    *,
    stage_name: str,
    id2label: dict[int, str],
    dataset_index: int,
) -> Image.Image:
    image = get_image(clone_example(example), convert_mode="RGB")
    canvas = image.copy()
    font = ImageFont.load_default()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))

    objects = get_object_records(example)
    for object_index, object_record in enumerate(objects):
        color = COLORS[object_index % len(COLORS)]
        rgb = hex_to_rgb(color)
        mask = object_record.get("mask")
        if isinstance(mask, np.ndarray) and mask.shape == (canvas.height, canvas.width):
            mask_rgba = np.zeros((canvas.height, canvas.width, 4), dtype=np.uint8)
            mask_rgba[mask.astype(bool)] = (*rgb, MASK_ALPHA)
            overlay = Image.alpha_composite(overlay, Image.fromarray(mask_rgba, mode="RGBA"))

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    for object_index, object_record in enumerate(objects):
        x, y, width, height = [float(value) for value in object_record["bbox"]]
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(canvas.width), x + width)
        y2 = min(float(canvas.height), y + height)
        color = COLORS[object_index % len(COLORS)]
        line_width = max(2, canvas.width // 300)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        category_id = int(object_record["category"])
        label = id2label.get(category_id, str(category_id))
        text = f"{object_index}: {label}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top
        text_x = x1
        text_y = max(0.0, y1 - text_height - 6)
        draw.rectangle((text_x, text_y, text_x + text_width + 8, text_y + text_height + 4), fill=color)
        draw.text((text_x + 4, text_y + 2), text, fill="white", font=font)

    header = (
        f"idx={dataset_index} stage={stage_name} "
        f"size={canvas.width}x{canvas.height} objects={len(objects)}"
    )
    left, top, right, bottom = draw.textbbox((0, 0), header, font=font)
    header_width = right - left
    header_height = bottom - top
    draw.rectangle((0, 0, header_width + 12, header_height + 8), fill="black")
    draw.text((6, 4), header, fill="white", font=font)

    footer = (
        f"image_id={example.get('image_id', '?')} "
        f"modified={bool(example.get('modified', False))}"
    )
    left, top, right, bottom = draw.textbbox((0, 0), footer, font=font)
    footer_width = right - left
    footer_height = bottom - top
    footer_y = max(0, canvas.height - footer_height - 8)
    draw.rectangle((0, footer_y - 2, footer_width + 12, footer_y + footer_height + 6), fill="black")
    draw.text((6, footer_y), footer, fill="white", font=font)

    return canvas


def make_stage_strip(images: list[Image.Image], *, tile_size: int) -> Image.Image:
    if not images:
        raise ValueError("Expected at least one rendered stage image.")

    strip = Image.new("RGB", (len(images) * tile_size, tile_size), color=(24, 24, 24))
    for index, image in enumerate(images):
        tile = image.copy()
        tile.thumbnail((tile_size, tile_size))
        offset_x = index * tile_size + (tile_size - tile.width) // 2
        offset_y = (tile_size - tile.height) // 2
        strip.paste(tile, (offset_x, offset_y))
    return strip


def stack_strips(strips: list[Image.Image], *, gap: int = 16) -> Image.Image:
    if not strips:
        raise ValueError("Expected at least one strip image.")

    width = max(strip.width for strip in strips)
    height = sum(strip.height for strip in strips) + gap * max(len(strips) - 1, 0)
    sheet = Image.new("RGB", (width, height), color=(16, 16, 16))

    offset_y = 0
    for strip in strips:
        sheet.paste(strip, ((width - strip.width) // 2, offset_y))
        offset_y += strip.height + gap

    return sheet


def save_visualization_set(
    prepared_dataset: PreparedDataset,
    split_name: str,
    *,
    pipeline_name: str,
    steps: list[AugmentationStep],
    id2label: dict[int, str],
    indices: list[int],
    output_dir: Path,
    tile_size: int,
    save_stage_images: bool,
    deterministic: bool,
    seed: int,
) -> dict[str, Any]:
    strips: list[Image.Image] = []
    sample_paths: list[str] = []

    pipeline_dir = output_dir / pipeline_name
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    for dataset_index in indices:
        snapshots = capture_pipeline_stages(
            prepared_dataset,
            dataset_index,
            steps=steps,
            deterministic=deterministic,
            seed=seed,
        )
        rendered_stages = [
            render_stage(
                snapshot,
                stage_name=stage_name,
                id2label=id2label,
                dataset_index=dataset_index,
            )
            for stage_name, snapshot in snapshots
        ]

        sample_dir = pipeline_dir / f"{split_name}_{dataset_index:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        if save_stage_images:
            for stage_index, ((stage_name, _), rendered) in enumerate(zip(snapshots, rendered_stages)):
                stage_slug = slugify(stage_name)
                rendered.save(sample_dir / f"{stage_index:02d}_{stage_slug}.jpg", quality=95)

        strip = make_stage_strip(rendered_stages, tile_size=tile_size)
        strip_path = sample_dir / "stages_strip.jpg"
        strip.save(strip_path, quality=95)
        strips.append(strip)
        sample_paths.append(str(strip_path))

    summary = stack_strips(strips, gap=max(8, tile_size // 24))
    summary_path = pipeline_dir / f"{split_name}_summary.jpg"
    summary.save(summary_path, quality=95)

    return {
        "pipeline": pipeline_name,
        "split": split_name,
        "num_samples": len(indices),
        "indices": indices,
        "output_dir": str(pipeline_dir),
        "summary": str(summary_path),
        "sample_strips": sample_paths,
    }


def main() -> None:
    args = parse_args()
    dataset = dataset_from_args(args)
    loader = build_detection_loader()

    requested_pipelines = []
    if args.pipeline in ("train", "both"):
        requested_pipelines.append(("train", args.train_split, build_train_augmentation_steps(args)))
    if args.pipeline in ("validation", "both"):
        requested_pipelines.append(("validation", args.validation_split, build_validation_augmentation_steps(args)))

    if not requested_pipelines:
        raise ValueError("No visualization pipelines were selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for pipeline_name, split_name, steps in requested_pipelines:
        if split_name not in dataset:
            raise KeyError(f"Split '{split_name}' not found in dataset. Available splits: {list(dataset.keys())}")

        split_dataset = dataset[split_name]
        indices = select_indices(
            split_dataset,
            indices=args.indices,
            num_samples=args.num_samples,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        if not indices:
            raise ValueError(f"No indices selected for split '{split_name}'.")

        _, id2label = load_label_metadata(dataset, split_name)
        prepared_dataset = PreparedDataset(split_dataset, loader=loader)
        result = save_visualization_set(
            prepared_dataset,
            split_name,
            pipeline_name=pipeline_name,
            steps=steps,
            id2label=id2label,
            indices=indices,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            save_stage_images=args.save_stage_images,
            deterministic=args.deterministic,
            seed=args.seed,
        )
        print("visualized_augmentations", result)


if __name__ == "__main__":
    main()
