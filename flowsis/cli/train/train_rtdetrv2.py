import argparse
import sys
from pathlib import Path
from typing import Any, cast

import torch
from datasets import ClassLabel, Dataset, DatasetDict, Image as HFImage, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from flowsis.rtdetrv2 import RTDetrV2
from flowsis.utils import (
    AugmentationPipeline,
    TransformDataset,
    build_autocast_context,
    build_grad_scaler,
    get_device,
    load_training_state,
    resolve_resume_checkpoint,
    save_checkpoint,
    set_seed,
    roi_square,
    rotation_augment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETRv2 on the HF detection dataset used by FlowSIS.")
    parser.add_argument("--dataset_path", type=str, default="data/dataset")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--validation_split", type=str, default="validation")
    parser.add_argument("--output_dir", type=str, default="outputs/rtdetrv2")
    parser.add_argument("--model_name_or_path", type=str, default="PekingU/rtdetr_v2_r18vd")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_every_epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--overfit_single_batch", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--sanity_decode", action="store_true")
    parser.add_argument("--run_inference_example", action="store_true")
    parser.add_argument("--inference_split", type=str, default="validation")
    parser.add_argument("--inference_index", type=int, default=0)
    parser.add_argument("--score_threshold", type=float, default=0.3)
    return parser.parse_args()


def log_event(name: str, payload: dict[str, Any]) -> None:
    print(name, payload)


def load_detection_dataset(args: argparse.Namespace) -> DatasetDict | IterableDatasetDict:
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        dataset = load_from_disk(str(dataset_path))

    if isinstance(dataset, (Dataset, IterableDataset)):
        raise ValueError("Expected a dataset with named splits, but received a single split dataset.")
    return dataset


def ensure_split_exists(dataset: DatasetDict | IterableDatasetDict, split_name: str, *, role: str) -> None:
    if split_name not in dataset:
        raise KeyError(f"Missing {role} split '{split_name}' in dataset.")


def dataset_to_annotation(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": int(example["image_id"]),
        "annotations": [
            {
                "bbox": [float(value) for value in bbox],
                "category_id": int(category_id),
                "area": float(area),
                "iscrowd": 0,
            }
            for bbox, category_id, area in zip(
                example["objects"]["bbox"],
                example["objects"]["category"],
                example["objects"]["area"],
            )
        ],
    }


def collate_examples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [example["image"].convert("RGB") for example in batch],
        "annotations": [dataset_to_annotation(example) for example in batch],
        "image_ids": [int(example["image_id"]) for example in batch],
        "orig_sizes": [(int(example["height"]), int(example["width"])) for example in batch],
    }


def load_label_metadata(
    dataset: DatasetDict | IterableDatasetDict,
    split_name: str,
) -> tuple[int, dict[int, str]]:
    category_feature = dataset[split_name].features["objects"]["category"]
    if hasattr(category_feature, "feature"):
        category_feature = category_feature.feature
    if not isinstance(category_feature, ClassLabel):
        raise TypeError(
            "Expected dataset feature objects.category to be a datasets.ClassLabel with contiguous 0-based ids."
        )

    id2label = {
        index: name if name else f"class_{index}"
        for index, name in enumerate(category_feature.names)
    }
    return category_feature.num_classes, id2label


def build_dataloader(
    split_dataset: Dataset | IterableDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    shuffle_batches = shuffle and not isinstance(split_dataset, IterableDataset)
    return DataLoader(
        split_dataset,
        batch_size=batch_size,
        shuffle=shuffle_batches,
        num_workers=num_workers,
        collate_fn=collate_examples,
        generator=generator,
    )


def estimate_total_steps(
    train_loader: DataLoader,
    *,
    epochs: int,
    max_steps: int | None,
    overfit_single_batch: bool,
) -> int:
    if max_steps is not None:
        return max_steps
    if overfit_single_batch:
        return epochs
    return epochs * len(train_loader)


def average_loss_dict(loss_sums: dict[str, float], count: int) -> dict[str, float]:
    if count == 0:
        return {}
    return {key: value / count for key, value in loss_sums.items()}


def sanity_decode_dataset(
    dataset: DatasetDict | IterableDatasetDict,
    *,
    split_names: list[str],
    max_samples: int = 2,
) -> None:
    checked = 0

    for split_name in split_names:
        if split_name not in dataset:
            continue

        split_dataset = dataset[split_name]
        raw_split = split_dataset.cast_column("image", HFImage(decode=False))
        limit = min(len(split_dataset), max_samples - checked)

        for index in range(limit):
            raw_image = raw_split[index]["image"]
            image_source = raw_image.get("path") if isinstance(raw_image, dict) else raw_image

            try:
                image = split_dataset[index]["image"].convert("RGB")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to decode dataset image for split='{split_name}' index={index} source={image_source!r}: {exc}"
                ) from exc

            log_event(
                "sanity_decode",
                {
                    "split": split_name,
                    "index": index,
                    "source": image_source,
                    "size": image.size,
                    "mode": image.mode,
                },
            )
            checked += 1
            if checked >= max_samples:
                return

    if checked == 0:
        raise ValueError(f"No decodable samples found in splits: {split_names}")


def build_model(
    *,
    model_name_or_path: str,
    resume_checkpoint: Path | None,
    num_labels: int,
    id2label: dict[int, str],
    device: torch.device,
) -> RTDetrV2:
    if resume_checkpoint is not None:
        return RTDetrV2.from_pretrained(str(resume_checkpoint), device=device)

    return RTDetrV2(
        model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        device=device,
    )


def build_optimizer(model: RTDetrV2, *, lr: float, weight_decay: float) -> AdamW:
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


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


def get_epoch_batches(
    train_loader: DataLoader,
    *,
    overfit_single_batch: bool,
    max_steps: int | None,
    global_step: int,
) -> list[dict[str, Any]] | DataLoader:
    if not overfit_single_batch:
        return train_loader

    single_batch = next(iter(train_loader))
    repeated_steps = 1 if max_steps is None else max(max_steps - global_step, 1)
    return [single_batch] * repeated_steps


def train_one_epoch(
    model: RTDetrV2,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    global_step: int,
    image_size: int,
    max_steps: int | None,
    overfit_single_batch: bool,
    device: torch.device,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[dict[str, Any], int]:
    model.train()
    epoch_loss_sum = 0.0
    epoch_batch_count = 0
    epoch_loss_dict_sums: dict[str, float] = {}
    first_loss: float | None = None
    last_loss: float | None = None

    optimizer.zero_grad(set_to_none=True)
    epoch_batches = get_epoch_batches(
        data_loader,
        overfit_single_batch=overfit_single_batch,
        max_steps=max_steps,
        global_step=global_step,
    )

    for batch in epoch_batches:
        if max_steps is not None and global_step >= max_steps:
            break

        with build_autocast_context(enabled=use_amp, device=device):
            forward_result = model(
                batch["images"],
                batch["annotations"],
                image_size=image_size,
                return_outputs=False,
            )
            if forward_result.loss is None:
                raise RuntimeError("Training forward did not return a loss.")
            loss = forward_result.loss

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

        global_step += 1
        epoch_loss_sum += loss_value
        epoch_batch_count += 1
        for key, value in forward_result.loss_dict.items():
            epoch_loss_dict_sums[key] = epoch_loss_dict_sums.get(key, 0.0) + float(value.detach().item())

        log_event(
            "train_step",
            {
                "epoch": epoch,
                "global_step": global_step,
                "loss": round(loss_value, 6),
            },
        )

    epoch_loss = epoch_loss_sum / max(epoch_batch_count, 1)
    epoch_loss_dict = average_loss_dict(epoch_loss_dict_sums, epoch_batch_count)
    summary = {
        "epoch": epoch,
        "global_step": global_step,
        "loss": round(epoch_loss, 6),
        "loss_dict": {key: round(value, 6) for key, value in epoch_loss_dict.items()},
        "first_loss": None if first_loss is None else round(first_loss, 6),
        "last_loss": None if last_loss is None else round(last_loss, 6),
    }
    return summary, global_step


def evaluate(
    model: RTDetrV2,
    data_loader: DataLoader,
    *,
    image_size: int,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    loss_sums: dict[str, float] = {}

    with torch.no_grad():
        for batch in data_loader:
            with build_autocast_context(enabled=use_amp, device=device):
                forward_result = model(
                    batch["images"],
                    batch["annotations"],
                    image_size=image_size,
                    return_outputs=False,
                )
            if forward_result.loss is None:
                continue

            total_loss += float(forward_result.loss.item())
            total_batches += 1
            for key, value in forward_result.loss_dict.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().item())

    return total_loss / max(total_batches, 1), average_loss_dict(loss_sums, total_batches)


def save_final_model(model: RTDetrV2, output_dir: Path) -> Path:
    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    return final_dir


def run_inference_example(
    checkpoint_dir: str | Path,
    dataset: DatasetDict | IterableDatasetDict,
    *,
    split: str,
    index: int,
    image_size: int,
    threshold: float,
    device: torch.device,
) -> None:
    model = RTDetrV2.from_pretrained(str(checkpoint_dir), device=device)
    sample = dataset[split][index]
    inference = model.infer(sample["image"].convert("RGB"), image_size=image_size, threshold=threshold)
    first_detection = inference.detections[0]
    log_event(
        "inference_example",
        {
            "checkpoint": str(checkpoint_dir),
            "num_detections": int(first_detection["scores"].numel()),
            "patch_tokens_shape": tuple(inference.encodings["patch_tokens"].shape),
            "image_embedding_shape": tuple(inference.encodings["image_embedding"].shape),
        },
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    dataset = load_detection_dataset(args)

    if args.sanity_decode:
        sanity_decode_dataset(
            dataset,
            split_names=[args.train_split, args.validation_split],
        )
        return

    ensure_split_exists(dataset, args.train_split, role="train")
    if args.validation_split not in dataset:
        log_event("validation_skip", {"missing_split": args.validation_split})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_labels, id2label = load_label_metadata(dataset, args.train_split)
    resume_checkpoint = resolve_resume_checkpoint(args.resume_from)
    model = build_model(
        model_name_or_path=args.model_name_or_path,
        resume_checkpoint=resume_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        device=device,
    )

    augments = [rotation_augment, roi_square]
    augment_kwargs = [
        {"pad": 1},
        {"image_size": 640},
    ]

    transform_dataset = TransformDataset(
        dataset[args.train_split],
        AugmentationPipeline(augments, augment_kwargs),
    )
    transform_dataset = cast(Dataset, transform_dataset) # To calm type checker on HF Dataset and torch Dataset

    train_loader = build_dataloader(
        split_dataset=transform_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=not args.overfit_single_batch,
        seed=args.seed,
    )
    
    validation_loader = None
    if args.validation_split in dataset:
        validation_loader = build_dataloader(
            dataset[args.validation_split],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            seed=args.seed,
        )

    total_steps = estimate_total_steps(
        train_loader,
        epochs=args.epochs,
        max_steps=args.max_steps,
        overfit_single_batch=args.overfit_single_batch,
    )
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )
    scaler = build_grad_scaler(enabled=args.amp, device=device)

    start_epoch = 0
    global_step = 0
    if resume_checkpoint is not None:
        resume_state = load_training_state(
            resume_checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location="cpu",
        )
        start_epoch = resume_state.epoch
        global_step = resume_state.global_step
        log_event(
            "resumed_from",
            {
                "checkpoint": str(resume_state.checkpoint_dir),
                "epoch": start_epoch,
                "global_step": global_step,
            },
        )

    first_train_loss: float | None = None
    last_train_loss: float | None = None

    for epoch in range(start_epoch, args.epochs):
        epoch_summary, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            epoch=epoch,
            global_step=global_step,
            image_size=args.image_size,
            max_steps=args.max_steps,
            overfit_single_batch=args.overfit_single_batch,
            device=device,
            use_amp=args.amp,
            scaler=scaler,
        )
        log_event("train_epoch", epoch_summary)

        if epoch_summary["first_loss"] is not None and first_train_loss is None:
            first_train_loss = float(epoch_summary["first_loss"])
        if epoch_summary["last_loss"] is not None:
            last_train_loss = float(epoch_summary["last_loss"])

        if validation_loader is not None:
            validation_loss, validation_loss_dict = evaluate(
                model,
                validation_loader,
                image_size=args.image_size,
                device=device,
                use_amp=args.amp,
            )
            log_event(
                "validation_epoch",
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": round(validation_loss, 6),
                    "loss_dict": {key: round(value, 6) for key, value in validation_loss_dict.items()},
                },
            )

        if (epoch + 1) % args.save_every_epochs == 0 or epoch == args.epochs - 1:
            checkpoint_dir = save_checkpoint(
                model,
                optimizer,
                scheduler,
                output_dir,
                epoch=epoch + 1,
                global_step=global_step,
                scaler=scaler,
            )
            log_event(
                "saved_checkpoint",
                {
                    "path": str(checkpoint_dir),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                },
            )

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    final_dir = save_final_model(model, output_dir)
    log_event("saved_final_model", {"path": str(final_dir)})

    if first_train_loss is not None and last_train_loss is not None:
        log_event(
            "train_summary",
            {
                "first_loss": round(first_train_loss, 6),
                "last_loss": round(last_train_loss, 6),
            },
        )

    if args.run_inference_example:
        ensure_split_exists(dataset, args.inference_split, role="inference")
        run_inference_example(
            final_dir,
            dataset,
            split=args.inference_split,
            index=args.inference_index,
            image_size=args.image_size,
            threshold=args.score_threshold,
            device=device,
        )


if __name__ == "__main__":
    main()
