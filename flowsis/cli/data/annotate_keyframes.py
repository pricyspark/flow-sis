import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from transformers import Sam3VideoModel, Sam3VideoProcessor

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
        out_mask_dir=Path(out_mask_dir_raw).expanduser().resolve() if out_mask_dir_raw else None,
        out_bbox_dir=Path(out_bbox_dir_raw).expanduser().resolve() if out_bbox_dir_raw else None,
        device=device,
    )
    
    
def collect_job(config: SessionConfig) -> AnnotationJob | None:
    print("\nNew annotation job")
    print("Enter 'q' to quit.\n")

    video_input = prompt_required("Video filename or path")

    if video_input.lower() in {"q", "quit", "exit"}:
        return None

    video_path = resolve_video_path(video_input, config.video_dir)

    text_prompt = prompt_required( # TODO: this isn't actually required if default text prompt is set
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
        processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
        outputs_per_frame[model_outputs.frame_idx] = processed_outputs
        
    return outputs_per_frame


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
            tile_masks(outputs_per_frame, job.tensor, 5)
            
            save = yes_no("\nSave? [Y/n]: ")
            if save:
                all_boxes = np.empty((len(outputs_per_frame), 4), dtype=np.int64)
                for i, output in outputs_per_frame.items():
                    mask = output["masks"][0].cpu().numpy()
                    f_stem = job.keyframes[i].stem
                    frame_idx = int(f_stem.partition('-')[2].partition('_')[0])
                    print(f"mask location {job.out_mask_dir / f"{frame_idx}.npz"}")
                    save_binary(job.out_mask_dir / f"{frame_idx}.npz", mask)
                    
                    xyxy = output["boxes"][0].cpu().numpy() # TODO: fix
                    xywh = xyxy.copy()
                    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
                    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
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
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def tile_masks(outputs_per_frame: dict, keyframe_batch: torch.Tensor, n_cols: int):
    plt.close("all")
    
    n_rows = int(math.ceil(len(outputs_per_frame) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.7, n_rows * 4.8), squeeze=False)
    
    axes_flat = axes.ravel()
    
    for i, ax in enumerate(axes_flat):
        if i >= len(outputs_per_frame):
            ax.axis("off")
            continue
        
        ax.imshow(keyframe_batch[i].cpu().numpy())
        ax.axis("off")
        try:
            show_mask(outputs_per_frame[i]["masks"].cpu().numpy(), ax)
        except Exception as e:
            print(i)
            print(e)
    plt.tight_layout()
    plt.show()
    
def save_binary(path, arr):
    if arr.dtype != np.bool_:
        raise TypeError
    
    packed = np.packbits(arr.ravel())
    np.savez_compressed(path, packed=packed, shape=arr.shape)
    
def get_keyframes(dir: Path):
    keyframes = [
        file
        for file in dir.iterdir()
        if file.suffix in IMAGE_EXTENSIONS
    ]
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
