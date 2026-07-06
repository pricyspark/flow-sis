import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from flowsis.base_head import BaseFusionHead
from flowsis.data import CallablePipeline, PreparedDataset, load_object_image
from flowsis.data.augment import center_square_augment, overlap_augment, photometric_augment, roi_square_augment, rotation_augment
from flowsis.data.images import get_image
from flowsis.data.masks import load_binary
from flowsis.pretrained import RTDetrV2
from flowsis.utils import (
    build_autocast_context,
    build_grad_scaler,
    get_device,
    load_training_state,
    resolve_resume_checkpoint,
    set_seed,
)


PhaseName = Literal["offline", "online"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the FlowSIS base fusion head with cached RT-DETRv2 features, "
            "online frozen RT-DETRv2 features, or a staged combination of both."
        ),
    )
    parser.add_argument("--dataset_path", type=str, default="data/segmentation-dataset")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--validation_split", type=str, default="validation")
    parser.add_argument("--output_dir", type=str, default="outputs/base")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--rtdetrv2_name_or_path", type=str, default="PekingU/rtdetr_v2_r18vd")
    parser.add_argument(
        "--train_stages",
        type=str,
        default="online:5",
        help="Comma-separated stage plan, e.g. 'offline:8,online:2'.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_every_epochs", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_rotation_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply rotation augmentation during online-image stages.",
    )
    parser.add_argument(
        "--use_roi_square_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply ROI square cropping during online-image stages.",
    )
    parser.add_argument(
        "--use_overlap_augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply overlap compositing during online-image stages.",
    )
    parser.add_argument(
        "--overlap_min_overlays",
        type=int,
        default=1,
        help="Minimum number of samples to composite when overlap augmentation is enabled.",
    )
    parser.add_argument(
        "--overlap_max_overlays",
        type=int,
        default=1,
        help="Maximum number of samples to composite when overlap augmentation is enabled.",
    )
    parser.add_argument(
        "--overlap_p",
        type=float,
        default=0.5,
        help="Geometric continuation parameter used to sample additional overlap layers.",
    )
    parser.add_argument(
        "--use_photometric_augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply photometric augmentation during online-image stages. "
            "Keep this disabled if you want online behavior to match an offline cache exactly."
        ),
    )
    parser.add_argument("--num_decode_layers", type=int, default=2)
    parser.add_argument("--decode_embed_dim", type=int, default=256)
    parser.add_argument("--image_dim", type=int, default=256)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--decode_ffn_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, choices=("gelu", "relu"), default="gelu")
    parser.add_argument("--num_feature_levels", type=int, default=3)
    parser.add_argument("--decode_pos_encode", type=str, choices=("none", "first", "second", "all"), default="first")
    parser.add_argument("--image_self_attention", type=str, choices=("GLOBAL", "WINDOW", "none"), default="GLOBAL")
    parser.add_argument("--decode_window_size", type=int, default=8)
    parser.add_argument("--use_shifted_windows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multiscale_merge", type=str, choices=("conv", "deformable", "none"), default="conv")
    parser.add_argument("--deformable_num_points", type=int, default=4)
    parser.add_argument("--deformable_offset_scale", type=float, default=2.0)
    parser.add_argument("--aggregator_dim", type=int, default=None)
    parser.add_argument("--channel_aggregation", type=str, choices=("none", "sigmoid", "softmax"), default="sigmoid")
    parser.add_argument("--mask_feature_source", type=str, choices=("merged", "highest_resolution"), default="merged")
    parser.add_argument("--mask_head_hidden_dim", type=int, default=None)
    parser.add_argument("--mask_output_dim", type=int, default=1)
    return parser.parse_args()


def log_event(name: str, payload: dict[str, Any]) -> None:
    print(name, payload)


def parse_stage_spec(spec: str) -> list[tuple[PhaseName, int]]:
    stages: list[tuple[PhaseName, int]] = []
    for item in spec.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        name, _, epochs_text = stripped.partition(":")
        phase = cast(PhaseName, name)
        if phase not in {"offline", "online"}:
            raise ValueError(f"Unsupported training stage {phase!r}.")
        epochs = int(epochs_text or "1")
        if epochs <= 0:
            raise ValueError(f"Stage epochs must be positive, but received {epochs}.")
        stages.append((phase, epochs))
    if not stages:
        raise ValueError("At least one training stage is required.")
    return stages


def load_segmentation_dataset(dataset_path: str | Path) -> DatasetDict:
    dataset = load_from_disk(str(dataset_path))
    if isinstance(dataset, Dataset):
        raise TypeError("Expected a DatasetDict for segmentation training.")
    return dataset


def ensure_split_exists(dataset: DatasetDict, split_name: str, *, role: str) -> None:
    if split_name not in dataset:
        raise KeyError(f"Missing {role} split '{split_name}' in dataset.")


def load_target_mask(example: dict[str, Any]) -> torch.Tensor:
    mask = load_binary(example["target_mask_path"]).astype("float32", copy=False)
    return torch.from_numpy(mask)


def load_segmentation_objects(example: dict[str, Any], **_: Any) -> dict[str, Any]:
    if "objects" not in example:
        return example
    for obj in example["objects"]:
        if "mask" in obj:
            continue
        if int(obj["video_id"]) != int(example["video_id"]) or int(obj["frame_idx"]) != int(example["frame_idx"]):
            continue
        obj["mask"] = load_binary(example["target_mask_path"])
    return example


def load_target_mask_from_objects(example: dict[str, Any]) -> torch.Tensor:
    target_video_id = int(example["video_id"])
    target_frame_idx = int(example["frame_idx"])
    for obj in example.get("objects", []):
        if int(obj["video_id"]) != target_video_id or int(obj["frame_idx"]) != target_frame_idx:
            continue
        if "mask" not in obj:
            continue
        mask = obj["mask"]
        return torch.from_numpy(mask.astype("float32", copy=False))
    return load_target_mask(example)


def resize_mask_tensor(mask: torch.Tensor, *, image_size: int) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask tensor, but received shape {tuple(mask.shape)}.")
    resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=(image_size, image_size),
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0)


def load_text_embedding(example: dict[str, Any]) -> torch.Tensor:
    text_embedding = torch.load(example["text_embedding_path"], map_location="cpu", weights_only=False)
    if not isinstance(text_embedding, torch.Tensor):
        raise TypeError(
            f"Expected text embedding tensor at {example['text_embedding_path']}, "
            f"received {type(text_embedding).__name__}."
        )
    if text_embedding.ndim == 3:
        return text_embedding.mean(dim=0)
    if text_embedding.ndim == 2:
        return text_embedding
    raise ValueError(
        f"Expected text embeddings to have shape [P,T,D] or [T,D], "
        f"but received {tuple(text_embedding.shape)}."
    )


def load_cached_feature_maps(cache_dir: str | Path) -> list[torch.Tensor]:
    cache_path = Path(cache_dir)
    bundle_path = cache_path / "feature_maps.pt"
    feature_maps: Any
    if bundle_path.exists():
        feature_maps = torch.load(bundle_path, map_location="cpu", weights_only=False)
        if isinstance(feature_maps, dict):
            feature_maps = feature_maps.get("feature_maps")
    else:
        level_paths = sorted(cache_path.glob("level_*.pt"))
        feature_maps = [torch.load(level_path, map_location="cpu", weights_only=False) for level_path in level_paths]

    if not isinstance(feature_maps, list) or not feature_maps:
        raise FileNotFoundError(
            f"Could not load cached multi-scale features from {cache_path}. "
            "Expected feature_maps.pt or level_*.pt files."
        )

    normalized: list[torch.Tensor] = []
    for level_index, feature_map in enumerate(feature_maps):
        if not isinstance(feature_map, torch.Tensor):
            raise TypeError(
                f"Expected cached feature tensor at level {level_index}, "
                f"received {type(feature_map).__name__}."
            )
        if feature_map.ndim == 4 and feature_map.shape[0] == 1:
            feature_map = feature_map.squeeze(0)
        if feature_map.ndim != 3:
            raise ValueError(
                f"Expected cached feature map shaped [C,H,W], but received {tuple(feature_map.shape)}."
            )
        normalized.append(feature_map.float())
    return normalized


def build_online_dataset(split_dataset: Dataset, args: argparse.Namespace) -> Dataset:
    loader = CallablePipeline((load_object_image, load_segmentation_objects))
    augmentations = []
    augmentation_kwargs = []

    if args.use_rotation_augment:
        augmentations.append(rotation_augment)
        augmentation_kwargs.append({"pad": 1})
    if args.use_roi_square_augment:
        augmentations.append(roi_square_augment)
        augmentation_kwargs.append({"crop_size": args.image_size})
    if args.use_overlap_augment:
        overlay_prepare = CallablePipeline(augmentations, augmentation_kwargs)
        augmentations.append(overlap_augment)
        augmentation_kwargs.append(
            {
                "min_overlays": args.overlap_min_overlays,
                "max_overlays": args.overlap_max_overlays,
                "p": args.overlap_p,
                "overlay_prepare": overlay_prepare,
            }
        )
    if args.use_photometric_augment:
        augmentations.append(photometric_augment)
        augmentation_kwargs.append({})

    if not augmentations:
        return cast(Dataset, PreparedDataset(split_dataset, loader=loader))

    return cast(
        Dataset,
        PreparedDataset(
            split_dataset,
            loader=loader,
            augment=CallablePipeline(augmentations, augmentation_kwargs),
        ),
    )


def build_validation_dataset(split_dataset: Dataset, args: argparse.Namespace) -> Dataset:
    return cast(
        Dataset,
        PreparedDataset(
            split_dataset,
            loader=CallablePipeline((load_object_image, load_segmentation_objects)),
            augment=CallablePipeline((center_square_augment,), ({"crop_size": args.image_size},)),
        ),
    )


def collate_online_examples(batch: list[dict[str, Any]], *, image_size: int) -> dict[str, Any]:
    images = [get_image(example, convert_mode="RGB") for example in batch]
    return {
        "images": images,
        "text_embeddings": torch.stack([load_text_embedding(example) for example in batch], dim=0),
        "target_masks": torch.stack(
            [resize_mask_tensor(load_target_mask_from_objects(example), image_size=image_size) for example in batch],
            dim=0,
        ),
        "mask_output_sizes": [(image_size, image_size) for _ in images],
        "cache_keys": [str(example["cache_key"]) for example in batch],
    }


def collate_offline_examples(batch: list[dict[str, Any]], *, image_size: int) -> dict[str, Any]:
    feature_lists = [load_cached_feature_maps(example["cache_dir"]) for example in batch]
    num_levels = len(feature_lists[0])
    if any(len(feature_list) != num_levels for feature_list in feature_lists):
        raise ValueError("All cached feature examples in a batch must have the same number of levels.")

    stacked_feature_levels = [
        torch.stack([feature_list[level_index] for feature_list in feature_lists], dim=0)
        for level_index in range(num_levels)
    ]

    return {
        "multi_image_features": stacked_feature_levels,
        "text_embeddings": torch.stack([load_text_embedding(example) for example in batch], dim=0),
        "target_masks": torch.stack(
            [resize_mask_tensor(load_target_mask(example), image_size=image_size) for example in batch],
            dim=0,
        ),
        "mask_output_sizes": [(image_size, image_size) for _ in batch],
        "cache_keys": [str(example["cache_key"]) for example in batch],
    }


def build_dataloader(
    split_dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    phase: PhaseName,
    image_size: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    if phase == "online":
        collate_fn = lambda batch: collate_online_examples(batch, image_size=image_size)
    else:
        collate_fn = lambda batch: collate_offline_examples(batch, image_size=image_size)
    return DataLoader(
        split_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
    )


def build_head(args: argparse.Namespace, device: torch.device) -> BaseFusionHead:
    head = BaseFusionHead(
        num_decode_layers=args.num_decode_layers,
        decode_embed_dim=args.decode_embed_dim,
        image_dim=args.image_dim,
        text_dim=args.text_dim,
        nhead=args.nhead,
        decode_ffn_dim=args.decode_ffn_dim,
        dropout=args.dropout,
        activation=cast(Literal["gelu", "relu"], args.activation),
        num_feature_levels=args.num_feature_levels,
        decode_pos_encode=cast(Literal["none", "first", "second", "all"], args.decode_pos_encode),
        image_self_attention=cast(Literal["GLOBAL", "WINDOW", "none"], args.image_self_attention),
        decode_window_size=args.decode_window_size,
        use_shifted_windows=args.use_shifted_windows,
        multiscale_merge=cast(Literal["conv", "deformable", "none"], args.multiscale_merge),
        deformable_num_points=args.deformable_num_points,
        deformable_offset_scale=args.deformable_offset_scale,
        aggregator_dim=args.aggregator_dim,
        channel_aggregation=cast(Literal["none", "sigmoid", "softmax"], args.channel_aggregation),
        mask_feature_source=cast(Literal["merged", "highest_resolution"], args.mask_feature_source),
        mask_head_hidden_dim=args.mask_head_hidden_dim,
        mask_output_dim=args.mask_output_dim,
    )
    return head.to(device)


def build_frozen_encoder(args: argparse.Namespace, device: torch.device) -> RTDetrV2:
    model = RTDetrV2.from_pretrained(args.rtdetrv2_name_or_path, device=device)
    model.eval()
    model.requires_grad_(False)
    return model


def extract_online_feature_maps(
    model: RTDetrV2,
    images: Iterable[Image.Image],
    *,
    image_size: int,
) -> list[torch.Tensor]:
    batch = model.preprocess(images, annotations=None, image_size=image_size)
    outputs = model.model(
        pixel_values=batch["pixel_values"],
        pixel_mask=batch.get("pixel_mask"),
    )
    encoder_feature_maps = cast(list[torch.Tensor], outputs.encoder_last_hidden_state)
    if not encoder_feature_maps:
        raise RuntimeError("RT-DETRv2 did not return encoder feature maps for online base training.")
    return [feature_map.float() for feature_map in encoder_feature_maps]


def compute_batch_loss(
    head: BaseFusionHead,
    batch: dict[str, Any],
    *,
    device: torch.device,
    online_encoder: RTDetrV2 | None,
    image_size: int,
    use_amp: bool,
) -> torch.Tensor:
    text_embeddings = batch["text_embeddings"].to(device)
    target_masks = batch["target_masks"].to(device)
    mask_output_size = target_masks.shape[-2:]

    if "multi_image_features" in batch:
        multi_image_features = [feature_level.to(device) for feature_level in batch["multi_image_features"]]
    else:
        if online_encoder is None:
            raise RuntimeError("Online stage requires a frozen RT-DETRv2 encoder.")
        with torch.no_grad():
            multi_image_features = extract_online_feature_maps(
                online_encoder,
                batch["images"],
                image_size=image_size,
            )
            multi_image_features = [feature_level.to(device) for feature_level in multi_image_features]

    with build_autocast_context(enabled=use_amp, device=device):
        outputs = head(
            multi_image_features,
            text_embeddings,
            mask_output_size=mask_output_size,
        )
        mask_logits = cast(torch.Tensor, outputs["mask_logits"])
        return F.binary_cross_entropy_with_logits(mask_logits, target_masks)


def build_optimizer(model: BaseFusionHead, *, lr: float, weight_decay: float) -> AdamW:
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def save_head_checkpoint(
    head: BaseFusionHead,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    *,
    epoch: int,
    global_step: int,
    scaler: torch.cuda.amp.GradScaler | None,
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{global_step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), checkpoint_dir / "model.pt")
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        checkpoint_dir / "training_state.pt",
    )
    (output_dir / "last_checkpoint").write_text(str(checkpoint_dir.resolve()))
    return checkpoint_dir


def load_head_checkpoint(
    head: BaseFusionHead,
    checkpoint_dir: Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[int, int]:
    model_path = checkpoint_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing base-head checkpoint weights: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    head.load_state_dict(state_dict)
    resume_state = load_training_state(
        checkpoint_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location="cpu",
    )
    return resume_state.epoch, resume_state.global_step


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    return get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_steps, 1),
    )


def estimate_total_steps(
    stages: list[tuple[PhaseName, int]],
    train_loaders: dict[PhaseName, DataLoader],
    *,
    max_steps: int | None,
) -> int:
    if max_steps is not None:
        return max_steps
    return sum(len(train_loaders[phase]) * epochs for phase, epochs in stages)


def train_one_epoch(
    head: BaseFusionHead,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    phase: PhaseName,
    global_step: int,
    max_steps: int | None,
    device: torch.device,
    image_size: int,
    online_encoder: RTDetrV2 | None,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[dict[str, Any], int]:
    head.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    batch_count = 0
    first_loss: float | None = None
    last_loss: float | None = None

    for batch in data_loader:
        if max_steps is not None and global_step >= max_steps:
            break

        loss = compute_batch_loss(
            head,
            batch,
            device=device,
            online_encoder=online_encoder,
            image_size=image_size,
            use_amp=use_amp,
        )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.item())
        if first_loss is None:
            first_loss = loss_value
        last_loss = loss_value
        total_loss += loss_value
        batch_count += 1
        global_step += 1

        log_event(
            "train_step",
            {
                "epoch": epoch,
                "phase": phase,
                "global_step": global_step,
                "loss": round(loss_value, 6),
            },
        )

    epoch_loss = total_loss / max(batch_count, 1)
    return (
        {
            "epoch": epoch,
            "phase": phase,
            "global_step": global_step,
            "loss": round(epoch_loss, 6),
            "first_loss": None if first_loss is None else round(first_loss, 6),
            "last_loss": None if last_loss is None else round(last_loss, 6),
        },
        global_step,
    )


@torch.no_grad()
def evaluate(
    head: BaseFusionHead,
    data_loader: DataLoader,
    *,
    phase: PhaseName,
    device: torch.device,
    image_size: int,
    online_encoder: RTDetrV2 | None,
    use_amp: bool,
) -> float:
    head.eval()
    total_loss = 0.0
    batch_count = 0

    for batch in data_loader:
        loss = compute_batch_loss(
            head,
            batch,
            device=device,
            online_encoder=online_encoder,
            image_size=image_size,
            use_amp=use_amp,
        )
        total_loss += float(loss.item())
        batch_count += 1

    average_loss = total_loss / max(batch_count, 1)
    log_event(
        "validation_epoch",
        {
            "phase": phase,
            "loss": round(average_loss, 6),
        },
    )
    return average_loss


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    dataset = load_segmentation_dataset(args.dataset_path)
    ensure_split_exists(dataset, args.train_split, role="train")
    ensure_split_exists(dataset, args.validation_split, role="validation")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = parse_stage_spec(args.train_stages)
    train_loaders: dict[PhaseName, DataLoader] = {}
    validation_loaders: dict[PhaseName, DataLoader] = {}

    if any(phase == "online" for phase, _ in stages):
        online_train_dataset = build_online_dataset(dataset[args.train_split], args)
        online_validation_dataset = build_validation_dataset(dataset[args.validation_split], args)
        train_loaders["online"] = build_dataloader(
            online_train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            seed=args.seed,
            phase="online",
            image_size=args.image_size,
        )
        validation_loaders["online"] = build_dataloader(
            online_validation_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            seed=args.seed,
            phase="online",
            image_size=args.image_size,
        )

    if any(phase == "offline" for phase, _ in stages):
        offline_train_dataset = cast(Dataset, dataset[args.train_split])
        offline_validation_dataset = cast(Dataset, dataset[args.validation_split])
        train_loaders["offline"] = build_dataloader(
            offline_train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            seed=args.seed,
            phase="offline",
            image_size=args.image_size,
        )
        validation_loaders["offline"] = build_dataloader(
            offline_validation_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            seed=args.seed,
            phase="offline",
            image_size=args.image_size,
        )

    head = build_head(args, device)
    optimizer = build_optimizer(head, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = estimate_total_steps(stages, train_loaders, max_steps=args.max_steps)
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)
    scaler = build_grad_scaler(enabled=args.amp, device=device)

    start_epoch = 0
    global_step = 0
    resume_checkpoint = resolve_resume_checkpoint(args.resume_from)
    if resume_checkpoint is not None:
        start_epoch, global_step = load_head_checkpoint(
            head,
            resume_checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        log_event(
            "resumed_from",
            {
                "checkpoint": str(resume_checkpoint),
                "epoch": start_epoch,
                "global_step": global_step,
            },
        )

    online_encoder = build_frozen_encoder(args, device) if "online" in train_loaders else None

    epoch_index = start_epoch
    for phase, phase_epochs in stages:
        train_loader = train_loaders[phase]
        validation_loader = validation_loaders[phase]

        log_event(
            "train_phase",
            {
                "phase": phase,
                "epochs": phase_epochs,
                "num_batches": len(train_loader),
            },
        )

        for _ in range(phase_epochs):
            epoch_summary, global_step = train_one_epoch(
                head,
                train_loader,
                optimizer,
                scheduler,
                epoch=epoch_index,
                phase=phase,
                global_step=global_step,
                max_steps=args.max_steps,
                device=device,
                image_size=args.image_size,
                online_encoder=online_encoder if phase == "online" else None,
                use_amp=args.amp,
                scaler=scaler,
            )
            log_event("train_epoch", epoch_summary)

            evaluate(
                head,
                validation_loader,
                phase=phase,
                device=device,
                image_size=args.image_size,
                online_encoder=online_encoder if phase == "online" else None,
                use_amp=args.amp,
            )

            if (epoch_index + 1) % args.save_every_epochs == 0:
                checkpoint_dir = save_head_checkpoint(
                    head,
                    optimizer,
                    scheduler,
                    output_dir,
                    epoch=epoch_index + 1,
                    global_step=global_step,
                    scaler=scaler,
                )
                log_event(
                    "saved_checkpoint",
                    {
                        "path": str(checkpoint_dir),
                        "epoch": epoch_index + 1,
                        "global_step": global_step,
                        "phase": phase,
                    },
                )

            epoch_index += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), final_dir / "model.pt")
    log_event("saved_final_model", {"path": str(final_dir / 'model.pt')})


if __name__ == "__main__":
    main()
