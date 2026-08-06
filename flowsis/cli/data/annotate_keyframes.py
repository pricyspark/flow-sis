import math
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from transformers import Sam3VideoModel, Sam3VideoProcessor

from flowsis.data.masks import mask2xywh
from flowsis.utils import get_device

IMAGE_EXTENSIONS = Image.registered_extensions().keys()


@dataclass
class SessionConfig:
    model: Sam3VideoModel
    processor: Sam3VideoProcessor
    video_dir: Path | None = None
    text_prompt: str | None = None
    out_mask_dir: Path | None = None
    out_bbox_dir: Path | None = None
    device: torch.device = torch.device("cpu")


@dataclass
class AnnotationJob:
    video_path: Path
    keyframes: list[Path]
    text_prompt: str
    tensor: torch.Tensor
    out_mask_dir: Path
    out_bbox_dir: Path
    device: torch.device


def prompt_optional(message: str, default: str | None = None) -> str | None:
    if default:
        value = input(f"{message} [{default}]: ").strip()
        return value or default

    value = input(f"{message}: ").strip()
    return value or None


def prompt_required(message: str, default: str | None = None) -> str:
    while True:
        value = prompt_optional(message, default)
        if value:
            return value

        print("This value is required.")


def resolve_video_path(video_input: str, video_dir: Path | None) -> Path:
    path = Path(video_input).expanduser()

    # If the user enters a full or relative path, use it directly.
    if path.parent != Path("."):
        return path.resolve()

    # If the user only enters a filename, append it to video_dir.
    if video_dir is not None:
        return (video_dir / path).resolve()

    return path.resolve()


def collect_session_config() -> SessionConfig:
    device = get_device()
    print(f"Using device {device}")
    model = Sam3VideoModel.from_pretrained("facebook/sam3", device_map=device)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    print("\nSession setup")
    print("Leave optional values blank if you want to enter them per video.\n")

    text_prompt = prompt_optional("Default text prompt")
    video_dir_raw = prompt_optional("Video frame directory")
    out_mask_dir_raw = prompt_optional("Default output mask directory")
    out_bbox_dir_raw = prompt_optional("Default output bounding box directory")

    return SessionConfig(
        model=model,
        processor=processor,
        video_dir=Path(video_dir_raw).expanduser().resolve() if video_dir_raw else None,
        text_prompt=text_prompt,
        out_mask_dir=(
            Path(out_mask_dir_raw).expanduser().resolve() if out_mask_dir_raw else None
        ),
        out_bbox_dir=(
            Path(out_bbox_dir_raw).expanduser().resolve() if out_bbox_dir_raw else None
        ),
        device=device,
    )


def collect_job(config: SessionConfig) -> AnnotationJob | None:
    print("\nNew annotation job")
    print("Enter 'q' to quit.\n")

    video_input = prompt_required("Video filename or path")

    if video_input.lower() in {"q", "quit", "exit"}:
        return None

    video_path = resolve_video_path(video_input, config.video_dir)

    text_prompt = prompt_required(  # TODO: this isn't actually required if default text prompt is set
        "Text prompt",
        config.text_prompt,
    )

    out_mask_dir_raw = prompt_required(
        "Output mask directory",
        str(config.out_mask_dir) if config.out_mask_dir else None,
    )

    out_bbox_dir_raw = prompt_required(
        "Output bounding box directory",
        str(config.out_bbox_dir) if config.out_bbox_dir else None,
    )

    keyframes = get_keyframes(video_path)
    batch_np = load_keyframes(keyframes)
    batch_tensor = torch.tensor(batch_np, device=config.device)

    job = AnnotationJob(
        video_path=video_path,
        keyframes=keyframes,
        text_prompt=text_prompt,
        tensor=batch_tensor,
        out_mask_dir=Path(out_mask_dir_raw).expanduser().resolve(),
        out_bbox_dir=Path(out_bbox_dir_raw).expanduser().resolve(),
        device=config.device,
    )

    validate_job(job)
    return job


def validate_job(job: AnnotationJob) -> None:
    job.out_bbox_dir.mkdir(parents=True, exist_ok=True)
    job.out_mask_dir.mkdir(parents=True, exist_ok=True)


def annotate_video(
    model: Sam3VideoModel,
    processor: Sam3VideoProcessor,
    job: AnnotationJob,
) -> dict:
    print("\nRunning annotation")
    print(f"Video:        {job.video_path}")
    print(f"Text prompt:  {job.text_prompt}")
    print(f"Mask dir:     {job.out_mask_dir}")
    print(f"Bbox dir:     {job.out_bbox_dir}")

    inference_session = processor.init_video_session(
        video=job.tensor,
        inference_device=job.device,
        processing_device=job.device,
        video_storage_device=job.device,
    )

    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=job.text_prompt,
    )

    outputs_per_frame = {}
    # Pass show_progress_bar=True to display a tqdm progress bar.
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        show_progress_bar=True,
    ):
        processed_outputs = processor.postprocess_outputs(
            inference_session, model_outputs
        )
        outputs_per_frame[model_outputs.frame_idx] = processed_outputs

    return outputs_per_frame


def extract_output_masks(output: dict | None) -> np.ndarray:
    if output is None:
        return np.zeros((0, 0, 0), dtype=np.bool_)

    masks = output.get("masks")
    if masks is None:
        return np.zeros((0, 0, 0), dtype=np.bool_)

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    else:
        masks = np.asarray(masks)

    if masks.size == 0:
        return np.zeros((0, 0, 0), dtype=np.bool_)

    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    if masks.ndim != 3:
        raise ValueError(
            f"Expected masks with shape (N, H, W), received {masks.shape}."
        )

    return masks.astype(np.bool_, copy=False)


def merge_frame_annotation(
    output: dict | None,
    frame_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, int]:
    masks = extract_output_masks(output)
    height, width = frame_shape

    if len(masks) == 0:
        return np.zeros((height, width), dtype=np.bool_), np.zeros(4, dtype=np.int64), 0

    merged_mask = np.any(masks, axis=0)
    bbox = mask2xywh(merged_mask)
    if bbox is None:
        bbox_xywh = np.zeros(4, dtype=np.int64)
    else:
        bbox_xywh = np.asarray(bbox, dtype=np.int64)

    return merged_mask, bbox_xywh, len(masks)


def summarize_outputs(outputs_per_frame: dict, num_frames: int) -> tuple[int, int]:
    empty_frames = 0
    multi_mask_frames = 0
    for frame_idx in range(num_frames):
        num_masks = len(extract_output_masks(outputs_per_frame.get(frame_idx)))
        if num_masks == 0:
            empty_frames += 1
        elif num_masks > 1:
            multi_mask_frames += 1
    return empty_frames, multi_mask_frames


def parse_frame_number(frame_path: Path) -> int:
    match = re.search(r"frame-(\d+)", frame_path.stem)
    if match is None:
        raise ValueError(f"Unable to parse frame number from {frame_path.name}.")
    return int(match.group(1))


def yes_no(prompt) -> bool:
    while True:
        response = input(prompt).strip().lower()

        if response in {"", "y", "yes"}:
            return True

        if response in {"n", "no"}:
            return False

        print("Please enter y or n.")


def run_session() -> None:
    config = collect_session_config()

    while True:
        try:
            job = collect_job(config)
            if job is None:
                print("Exiting.")
                break

            outputs_per_frame = annotate_video(config.model, config.processor, job)
            empty_frames, multi_mask_frames = summarize_outputs(
                outputs_per_frame, len(job.keyframes)
            )
            print(
                "annotation_summary",
                {
                    "frames": len(job.keyframes),
                    "empty_frames": empty_frames,
                    "multi_mask_frames": multi_mask_frames,
                },
            )
            tile_masks(outputs_per_frame, job.tensor, 5, frame_paths=job.keyframes)

            save = yes_no("\nSave? [Y/n]: ")
            if save:
                all_boxes = np.zeros((len(job.keyframes), 4), dtype=np.int64)
                for i, frame_path in enumerate(job.keyframes):
                    output = outputs_per_frame.get(i)
                    mask, xywh, _ = merge_frame_annotation(
                        output,
                        frame_shape=tuple(job.tensor.shape[1:3]),
                    )
                    frame_idx = parse_frame_number(frame_path)
                    print(f"mask location {job.out_mask_dir / f"{frame_idx}.npz"}")
                    job.out_bbox_dir.mkdir(parents=True, exist_ok=True)
                    save_binary(
                        job.out_mask_dir / f"{frame_idx}.npz",
                        mask,
                    )
                    all_boxes[i] = xywh

                print(f"box location {job.out_bbox_dir / f"{job.video_path.stem}.npy"}")
                np.save(job.out_bbox_dir / f"{job.video_path.stem}.npy", all_boxes)

            again = yes_no("\nAnnotate another video? [Y/n]: ")
            if not again:
                print("Exiting.")
                break

        except Exception as exc:
            print(f"\nError: {exc}")

            retry = yes_no("Try another job? [Y/n]: ")
            if not retry:
                print("Exiting.")
                break


def show_mask(mask, ax, obj_id=None, random_color=False):
    mask = np.asarray(mask, dtype=np.bool_)
    if mask.ndim != 2 or mask.size == 0 or not mask.any():
        return

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    mask_image = mask[..., np.newaxis] * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_output_masks(output: dict | None, ax) -> int:
    masks = extract_output_masks(output)
    if len(masks) == 0:
        return 0

    object_ids = output.get("object_ids") if output is not None else None
    if isinstance(object_ids, torch.Tensor):
        object_ids = object_ids.detach().cpu().tolist()
    elif object_ids is None:
        object_ids = list(range(len(masks)))

    for mask, obj_id in zip(masks, object_ids):
        show_mask(mask, ax, obj_id=obj_id)

    return len(masks)


def resize_image_for_display(image: np.ndarray, max_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_size:
        return image

    scale = max_size / longest_side
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return np.asarray(
        Image.fromarray(image).resize(
            (resized_width, resized_height), resample=Image.Resampling.BILINEAR
        )
    )


def resize_masks_for_display(
    masks: np.ndarray, image_shape: tuple[int, int]
) -> np.ndarray:
    if len(masks) == 0:
        height, width = image_shape
        return np.zeros((0, height, width), dtype=np.bool_)

    resized_masks = []
    for mask in masks:
        mask_image = Image.fromarray(mask.astype(np.uint8, copy=False) * 255)
        resized_mask = mask_image.resize(
            (image_shape[1], image_shape[0]), resample=Image.Resampling.NEAREST
        )
        resized_masks.append(np.asarray(resized_mask) > 0)
    return np.stack(resized_masks, axis=0)


def draw_tile(
    ax,
    frame_idx: int,
    output: dict | None,
    keyframe_batch: torch.Tensor,
    max_frame_size: int,
    frame_paths: list[Path] | None,
) -> None:
    frame_image = keyframe_batch[frame_idx].cpu().numpy()
    display_image = resize_image_for_display(frame_image, max_frame_size)
    ax.imshow(display_image)
    ax.axis("off")

    masks = extract_output_masks(output)
    display_masks = resize_masks_for_display(masks, display_image.shape[:2])
    output_for_display = None if output is None else dict(output)
    if output_for_display is None:
        output_for_display = {"masks": display_masks}
    else:
        output_for_display["masks"] = display_masks

    num_masks = show_output_masks(output_for_display, ax)
    display_frame_number = frame_idx
    if frame_paths is not None:
        display_frame_number = parse_frame_number(frame_paths[frame_idx])

    ax.text(
        0.03,
        0.97,
        f"{display_frame_number} | {num_masks}",
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
    )
    if num_masks == 0:
        ax.text(
            0.5,
            0.5,
            "0 masks",
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            transform=ax.transAxes,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        )


def tile_masks(
    outputs_per_frame: dict,
    keyframe_batch: torch.Tensor,
    n_cols: int,
    max_frame_size: int = 512,
    frame_paths: list[Path] | None = None,
    max_rows_per_figure: int = 4,
):
    plt.close("all")

    num_frames = int(keyframe_batch.shape[0])
    if num_frames == 0:
        return

    frames_per_figure = max(1, n_cols * max_rows_per_figure)
    sample_image = resize_image_for_display(
        keyframe_batch[0].cpu().numpy(), max_frame_size
    )
    sample_height, sample_width = sample_image.shape[:2]
    tile_width = 2.4
    tile_height = max(1.8, tile_width * (sample_height / sample_width))

    for start_idx in range(0, num_frames, frames_per_figure):
        stop_idx = min(start_idx + frames_per_figure, num_frames)
        page_frame_count = stop_idx - start_idx
        n_rows = max(1, int(math.ceil(page_frame_count / n_cols)))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * tile_width, n_rows * tile_height),
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.04, hspace=0.08
        )

        axes_flat = axes.ravel()
        for page_offset, ax in enumerate(axes_flat):
            frame_idx = start_idx + page_offset
            if frame_idx >= stop_idx:
                ax.axis("off")
                continue

            draw_tile(
                ax,
                frame_idx=frame_idx,
                output=outputs_per_frame.get(frame_idx),
                keyframe_batch=keyframe_batch,
                max_frame_size=max_frame_size,
                frame_paths=frame_paths,
            )

        plt.show()


def save_binary(path, arr):
    if arr.dtype != np.bool_:
        raise TypeError

    packed = np.packbits(arr.ravel())
    np.savez_compressed(path, packed=packed, shape=arr.shape)


def get_keyframes(dir: Path):
    keyframes = [file for file in dir.iterdir() if file.suffix in IMAGE_EXTENSIONS]
    keyframes.sort(key=lambda path: path.stem.partition("_c")[0])
    return keyframes


def load_keyframes(keyframes):
    imgs = []
    for frame_path in keyframes:
        img = Image.open(frame_path).convert("RGB")
        arr = np.array(img)
        imgs.append(arr)
    batch = np.stack(imgs, axis=0)
    return batch


def main():
    run_session()


if __name__ == "__main__":
    main()
