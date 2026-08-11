import time
import argparse
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from flowsis.base_head import BaseFusionHead
from flowsis.artifacts import atomic_write_text
from flowsis.data import (
    CallablePipeline,
    PreparedDataset,
    load_feature_bundle,
    load_object_image,
)
from flowsis.data.augment import (
    center_square_augment,
    overlap_augment,
    photometric_augment,
    roi_square_augment,
    rotation_augment,
)
from flowsis.data.images import get_image
from flowsis.data.masks import load_binary
from flowsis.head_checkpoint import (
    load_head_checkpoint as load_head_bundle,
    save_head_checkpoint as save_head_bundle,
)
from flowsis.pretrained import (
    Detector,
    DetectorInferenceResult,
    load_detector,
)
from flowsis.pretrained.image_processing import image_to_rgb_tensor
from flowsis.cli.common import (
    add_detector_arguments,
    append_log_event,
    dataset_from_args,
    ensure_split_exists,
    log_event,
)
from flowsis.cli.train.training_manifest import write_run_manifest
from flowsis.utils import (
    build_autocast_context,
    build_grad_scaler,
    get_device,
    load_training_state,
    resolve_resume_checkpoint,
    save_training_state,
    set_seed,
)

PhaseName = Literal["offline", "online"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the FlowSIS base fusion head with cached detector features, "
            "online frozen detector features, or a staged combination of both."
        ),
    )
    parser.add_argument("--dataset-path", type=str, default="data/segmentation-dataset")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--validation-split", type=str, default="validation")
    parser.add_argument("--output-dir", type=str, default="outputs/base")
    parser.add_argument("--resume-from", type=str, default=None)
    add_detector_arguments(
        parser,
        model_flag="--detector-model",
        model_dest="detector_model_source",
    )
    parser.add_argument(
        "--train-stages",
        type=str,
        default="online:1",
        help="Comma-separated stage plan, e.g. 'offline:8,online:2'.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument(
        "--save-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save epoch-level events to OUTPUT_DIR/training_log.jsonl.",
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use, e.g. 'cuda:1' or 'cpu'. Defaults to automatic selection.",
    )
    parser.add_argument(
        "--use-rotation-augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply rotation augmentation during online-image stages.",
    )
    parser.add_argument(
        "--use-roi-square-augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply ROI square cropping during online-image stages.",
    )
    parser.add_argument(
        "--use-overlap-augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply overlap compositing during online-image stages.",
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
        default=True,
        help=(
            "Apply photometric augmentation during online-image stages. "
            "Keep this disabled if you want online behavior to match an offline cache exactly."
        ),
    )
    parser.add_argument("--num-decode-layers", type=int, default=1)
    parser.add_argument("--decode-embed-dim", type=int, default=128)
    parser.add_argument("--image-dim", type=int, default=256)
    parser.add_argument("--text-dim", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--decode-ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--activation", type=str, choices=("gelu", "relu"), default="gelu"
    )
    parser.add_argument("--num-feature-levels", type=int, default=3)
    parser.add_argument(
        "--decode-pos-encode",
        choices=("none", "first", "second", "all"),
        default="first",
    )
    parser.add_argument(
        "--image-self-attention",
        choices=("GLOBAL", "WINDOW", "none"),
        default="WINDOW",
    )
    parser.add_argument("--decode-window-size", type=int, default=8)
    parser.add_argument(
        "--use-shifted-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--multiscale-merge",
        choices=("conv", "deformable", "none"),
        default="conv",
    )
    parser.add_argument(
        "--conv-merge-refinement",
        choices=("standard", "depthwise"),
        default="depthwise",
    )
    parser.add_argument("--deformable-num-points", type=int, default=4)
    parser.add_argument("--deformable-offset-scale", type=float, default=2.0)
    parser.add_argument("--aggregator-dim", type=int, default=None)
    parser.add_argument(
        "--use-detector-query",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Condition each mask on its matched detector decoder query.",
    )
    parser.add_argument(
        "--detector-query-dim",
        type=int,
        default=256,
        help="Channel dimension of the detector's final decoder queries.",
    )
    parser.add_argument(
        "--channel-aggregation",
        choices=("none", "sigmoid", "softmax"),
        default="none",
    )
    parser.add_argument(
        "--mask-feature-source",
        choices=("merged", "highest_resolution"),
        default="merged",
    )
    parser.add_argument("--mask-head-hidden-dim", type=int, default=None)
    parser.add_argument("--mask-output-dim", type=int, default=1)
    parser.add_argument(
        "--mask-upsample-scales",
        type=int,
        nargs="+",
        default=(2, 2),
    )
    parser.add_argument(
        "--mask-convolution",
        choices=("standard", "depthwise_separable"),
        default="depthwise_separable",
    )
    parser.add_argument("--bce-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-smooth", type=float, default=1.0)
    parser.add_argument(
        "--prompt-dropout",
        type=float,
        default=0.2,
        help="Probability of dropping each prompt vector during training; one is always kept.",
    )
    return parser.parse_args()


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


def load_segmentation_objects(example: dict[str, Any], **_: Any) -> dict[str, Any]:
    for obj in example["objects"]:
        if "mask" in obj:
            continue
        obj["mask"] = load_binary(obj["mask_path"])
    return example


def resize_mask_tensor(mask: torch.Tensor, *, image_size: int) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(
            f"Expected 2D mask tensor, but received shape {tuple(mask.shape)}."
        )
    if mask.shape == (image_size, image_size):
        return mask
    resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=(image_size, image_size),
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0)


def normalize_object_box(obj: dict[str, Any], example: dict[str, Any]) -> list[float]:
    x, y, width, height = (float(value) for value in obj["bbox"])
    image_width = max(float(example["width"]), 1.0)
    image_height = max(float(example["height"]), 1.0)
    return [
        x / image_width,
        y / image_height,
        (x + width) / image_width,
        (y + height) / image_height,
    ]


def load_text_embedding(obj: dict[str, Any]) -> torch.Tensor:
    text_embedding = torch.load(
        obj["text_embedding_path"], map_location="cpu", weights_only=False
    )
    if not isinstance(text_embedding, torch.Tensor):
        raise TypeError(
            f"Expected text embedding tensor at {obj['text_embedding_path']}, "
            f"received {type(text_embedding).__name__}."
        )
    if text_embedding.ndim == 2:
        return text_embedding
    if text_embedding.ndim == 3:
        raise ValueError(
            "Legacy token-level prompt embeddings are not supported because averaging "
            "them mixes unrelated token positions and includes padding. Regenerate "
            "data/manifests/text-embeddings with the prompt embedding command so each "
            "file has shape [num_prompts, text_dim]."
        )
    raise ValueError(
        f"Expected pooled prompt embeddings to have shape [P,D], "
        f"but received {tuple(text_embedding.shape)}."
    )


def load_cached_feature_maps(
    cache_dir: str | Path,
    *,
    image_size: int,
    expected_levels: int,
    expected_channels: int,
) -> list[torch.Tensor]:
    bundle = load_feature_bundle(
        cache_dir,
        expected_levels=expected_levels,
        expected_channels=expected_channels,
        expected_image_size=image_size,
    )
    return list(bundle.feature_maps)


def build_online_dataset(split_dataset: Dataset, args: argparse.Namespace) -> Dataset:
    loader = CallablePipeline((load_object_image, load_segmentation_objects))
    augmentations = []
    augmentation_kwargs = []

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
    if args.use_rotation_augment:
        augmentations.append(rotation_augment)
        augmentation_kwargs.append({"pad": 1})
    if args.use_roi_square_augment:
        augmentations.append(roi_square_augment)
        augmentation_kwargs.append({"crop_size": args.image_size})
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


def build_validation_dataset(
    split_dataset: Dataset, args: argparse.Namespace
) -> Dataset:
    return cast(
        Dataset,
        PreparedDataset(
            split_dataset,
            loader=CallablePipeline((load_object_image, load_segmentation_objects)),
            augment=CallablePipeline(
                (center_square_augment,), ({"crop_size": args.image_size},)
            ),
        ),
    )


def collate_online_examples(
    batch: list[dict[str, Any]], *, image_size: int
) -> dict[str, Any]:
    image_tensors = [
        image_to_rgb_tensor(get_image(example, convert_mode="RGB")) for example in batch
    ]
    images: torch.Tensor | list[torch.Tensor]
    if all(image.shape == image_tensors[0].shape for image in image_tensors):
        images = torch.stack(image_tensors)
    else:
        images = image_tensors
    object_records = [
        (image_index, example, obj)
        for image_index, example in enumerate(batch)
        for obj in example["objects"]
        if "mask" in obj
    ]
    if not object_records:
        raise ValueError(
            "An online segmentation batch must contain at least one valid object."
        )
    return {
        "images": images,
        "object_image_indices": torch.tensor(
            [image_index for image_index, _, _ in object_records], dtype=torch.long
        ),
        "text_embeddings": torch.stack(
            [load_text_embedding(obj) for _, _, obj in object_records], dim=0
        ),
        "target_masks": torch.stack(
            [
                resize_mask_tensor(
                    torch.from_numpy(obj["mask"].astype("float32", copy=False)),
                    image_size=image_size,
                )
                for _, _, obj in object_records
            ],
            dim=0,
        ),
        "object_boxes": torch.tensor(
            [normalize_object_box(obj, example) for _, example, obj in object_records],
            dtype=torch.float32,
        ),
        "object_labels": torch.tensor(
            [int(obj["category"]) for _, _, obj in object_records],
            dtype=torch.long,
        ),
        "cache_keys": [str(example["cache_key"]) for example in batch],
    }


def collate_offline_examples(
    batch: list[dict[str, Any]],
    *,
    image_size: int,
    expected_levels: int,
    expected_channels: int,
) -> dict[str, Any]:
    feature_lists = [
        load_cached_feature_maps(
            example["cache_dir"],
            image_size=image_size,
            expected_levels=expected_levels,
            expected_channels=expected_channels,
        )
        for example in batch
    ]
    num_levels = len(feature_lists[0])
    if any(len(feature_list) != num_levels for feature_list in feature_lists):
        raise ValueError(
            "All cached feature examples in a batch must have the same number "
            "of levels."
        )

    stacked_feature_levels = [
        torch.stack(
            [feature_list[level_index] for feature_list in feature_lists], dim=0
        )
        for level_index in range(num_levels)
    ]

    object_records = [
        (image_index, example, obj)
        for image_index, example in enumerate(batch)
        for obj in example["objects"]
    ]
    if not object_records:
        raise ValueError(
            "An offline segmentation batch must contain at least one object."
        )
    return {
        "multi_image_features": stacked_feature_levels,
        "object_image_indices": torch.tensor(
            [image_index for image_index, _, _ in object_records], dtype=torch.long
        ),
        "text_embeddings": torch.stack(
            [load_text_embedding(obj) for _, _, obj in object_records], dim=0
        ),
        "target_masks": torch.stack(
            [
                resize_mask_tensor(
                    torch.from_numpy(
                        load_binary(obj["mask_path"]).astype("float32", copy=False)
                    ),
                    image_size=image_size,
                )
                for _, _, obj in object_records
            ],
            dim=0,
        ),
        "object_boxes": torch.tensor(
            [normalize_object_box(obj, example) for _, example, obj in object_records],
            dtype=torch.float32,
        ),
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
    expected_levels: int,
    expected_channels: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    if phase == "online":
        collate_fn = partial(collate_online_examples, image_size=image_size)
    else:
        collate_fn = partial(
            collate_offline_examples,
            image_size=image_size,
            expected_levels=expected_levels,
            expected_channels=expected_channels,
        )
    return DataLoader(
        split_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def build_head(args: argparse.Namespace, device: torch.device) -> BaseFusionHead:
    config = {
        "num_decode_layers": args.num_decode_layers,
        "decode_embed_dim": args.decode_embed_dim,
        "image_dim": args.image_dim,
        "text_dim": args.text_dim,
        "nhead": args.nhead,
        "decode_ffn_dim": args.decode_ffn_dim,
        "dropout": args.dropout,
        "activation": args.activation,
        "num_feature_levels": args.num_feature_levels,
        "decode_pos_encode": args.decode_pos_encode,
        "image_self_attention": args.image_self_attention,
        "decode_window_size": args.decode_window_size,
        "use_shifted_windows": args.use_shifted_windows,
        "multiscale_merge": args.multiscale_merge,
        "deformable_num_points": args.deformable_num_points,
        "deformable_offset_scale": args.deformable_offset_scale,
        "query_dim": (
            args.detector_query_dim if args.use_detector_query else None
        ),
        "conv_merge_refinement": args.conv_merge_refinement,
        "aggregator_dim": args.aggregator_dim,
        "channel_aggregation": args.channel_aggregation,
        "mask_feature_source": args.mask_feature_source,
        "mask_head_hidden_dim": args.mask_head_hidden_dim,
        "mask_output_dim": args.mask_output_dim,
        "mask_upsample_scales": tuple(args.mask_upsample_scales),
        "mask_convolution": args.mask_convolution,
    }
    head = BaseFusionHead(**config)
    head.architecture_config = config
    return head.to(device)


def build_frozen_encoder(args: argparse.Namespace, device: torch.device) -> Detector:
    model = load_detector(
        args.detector_model_source,
        architecture=args.detector_architecture,
        image_size=args.image_size,
        device=device,
    )
    model.eval()
    model.requires_grad_(False)
    return model


def extract_online_detector_output(
    model: Detector,
    images: Any,
) -> DetectorInferenceResult:
    return model.infer(images, threshold=0.0, device_preprocess=True)


def _corners_to_center(boxes: torch.Tensor) -> torch.Tensor:
    top_left, bottom_right = boxes.split(2, dim=-1)
    size = bottom_right - top_left
    return torch.cat((top_left + 0.5 * size, size), dim=-1)


def _center_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes.split(2, dim=-1)
    half_size = 0.5 * size
    return torch.cat((center - half_size, center + half_size), dim=-1)


def _pairwise_generalized_iou(
    left_boxes: torch.Tensor,
    right_boxes: torch.Tensor,
) -> torch.Tensor:
    left_top_left, left_bottom_right = left_boxes[:, None].split(2, dim=-1)
    right_top_left, right_bottom_right = right_boxes[None].split(2, dim=-1)

    intersection_top_left = torch.maximum(left_top_left, right_top_left)
    intersection_bottom_right = torch.minimum(
        left_bottom_right,
        right_bottom_right,
    )
    intersection_size = (intersection_bottom_right - intersection_top_left).clamp_min(
        0.0
    )
    intersection = intersection_size.prod(dim=-1)

    left_area = (left_bottom_right - left_top_left).clamp_min(0.0).prod(dim=-1)
    right_area = (right_bottom_right - right_top_left).clamp_min(0.0).prod(dim=-1)
    union = left_area + right_area - intersection
    iou = intersection / union.clamp_min(1e-7)

    enclosing_top_left = torch.minimum(left_top_left, right_top_left)
    enclosing_bottom_right = torch.maximum(left_bottom_right, right_bottom_right)
    enclosing_area = (enclosing_bottom_right - enclosing_top_left).prod(dim=-1)
    return iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-7)


def match_object_queries(
    query_logits: torch.Tensor,
    query_boxes: torch.Tensor,
    object_image_indices: torch.Tensor,
    object_labels: torch.Tensor,
    object_boxes: torch.Tensor,
    *,
    class_weight: float = 2.0,
    bbox_weight: float = 5.0,
    giou_weight: float = 2.0,
) -> torch.Tensor:
    """Match each mask target to one frozen detector query per image."""
    if query_logits.ndim != 3 or query_boxes.ndim != 3:
        raise ValueError("Expected batched detector query logits and boxes.")
    if query_logits.shape[:2] != query_boxes.shape[:2]:
        raise ValueError("Detector query logits and boxes must share [B,Q].")
    if query_boxes.shape[-1] != 4 or object_boxes.ndim != 2:
        raise ValueError("Detector and object boxes must have four coordinates.")
    num_objects = object_boxes.shape[0]
    if object_image_indices.shape != (num_objects,) or object_labels.shape != (
        num_objects,
    ):
        raise ValueError("Object indices, labels, and boxes must have matching rows.")
    if num_objects == 0:
        return torch.empty(0, dtype=torch.long, device=query_boxes.device)
    if object_labels.min() < 0 or object_labels.max() >= query_logits.shape[-1]:
        raise ValueError("Object label is outside the detector query-logit range.")

    matched_queries = torch.full(
        (num_objects,),
        -1,
        dtype=torch.long,
        device=query_boxes.device,
    )
    predicted_corners = _center_to_corners(query_boxes.float())
    target_centers = _corners_to_center(object_boxes.float())

    for image_index in object_image_indices.unique(sorted=True):
        target_indices = torch.nonzero(
            object_image_indices == image_index,
            as_tuple=False,
        ).flatten()
        image = int(image_index.item())
        labels = object_labels[target_indices]
        class_cost = -query_logits[image].float().sigmoid()[:, labels].transpose(0, 1)
        bbox_cost = torch.cdist(
            target_centers[target_indices],
            query_boxes[image].float(),
            p=1,
        )
        giou_cost = -_pairwise_generalized_iou(
            object_boxes[target_indices].float(),
            predicted_corners[image],
        )
        cost = (
            class_weight * class_cost
            + bbox_weight * bbox_cost
            + giou_weight * giou_cost
        )
        target_rows, query_columns = linear_sum_assignment(
            cost.detach().cpu().numpy()
        )
        assigned_targets = target_indices[
            torch.as_tensor(target_rows, device=target_indices.device)
        ]
        matched_queries[assigned_targets] = torch.as_tensor(
            query_columns,
            device=matched_queries.device,
        )

    if (matched_queries < 0).any():
        raise RuntimeError("Failed to match every segmentation object to a query.")
    return matched_queries


def compute_segmentation_objective(
    mask_logits: torch.Tensor,
    target_masks: torch.Tensor,
    object_image_indices: torch.Tensor | None = None,
    *,
    bce_weight: float,
    dice_weight: float,
    dice_smooth: float,
) -> dict[str, torch.Tensor]:
    def reduce_objects(values: torch.Tensor) -> torch.Tensor:
        if object_image_indices is None:
            return values.mean()
        num_images = int(object_image_indices.max().item()) + 1
        totals = values.new_zeros(num_images)
        counts = values.new_zeros(num_images)
        totals.scatter_add_(0, object_image_indices, values)
        counts.scatter_add_(0, object_image_indices, torch.ones_like(values))
        return (totals / counts.clamp_min(1.0))[counts > 0].mean()

    reduce_dims = tuple(range(1, mask_logits.ndim))
    bce_per_object = F.binary_cross_entropy_with_logits(
        mask_logits, target_masks, reduction="none"
    ).mean(dim=reduce_dims)
    bce = reduce_objects(bce_per_object)
    probabilities = mask_logits.sigmoid()
    intersection = (probabilities * target_masks).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + target_masks.sum(dim=reduce_dims)
    soft_dice = reduce_objects(
        (2.0 * intersection + dice_smooth) / (denominator + dice_smooth)
    )
    dice_loss = 1.0 - soft_dice
    loss = bce_weight * bce + dice_weight * dice_loss

    with torch.no_grad():
        prediction = probabilities >= 0.5
        target = target_masks >= 0.5
        hard_intersection = (prediction & target).sum(dim=reduce_dims).float()
        prediction_area = prediction.sum(dim=reduce_dims).float()
        target_area = target.sum(dim=reduce_dims).float()
        union = (prediction | target).sum(dim=reduce_dims).float()
        hard_dice = reduce_objects(
            (2.0 * hard_intersection + 1.0) / (prediction_area + target_area + 1.0)
        )
        iou = reduce_objects((hard_intersection + 1.0) / (union + 1.0))
        brier = reduce_objects(
            F.mse_loss(
                probabilities.float(), target_masks.float(), reduction="none"
            ).mean(dim=reduce_dims)
        )

    return {
        "loss": loss,
        "bce": bce.detach(),
        "dice_loss": dice_loss.detach(),
        "dice": hard_dice,
        "iou": iou,
        "brier": brier,
    }


def compute_batch_loss(
    head: BaseFusionHead,
    batch: dict[str, Any],
    *,
    online_encoder: Detector | None,
    use_amp: bool,
    bce_weight: float,
    dice_weight: float,
    dice_smooth: float,
    prompt_dropout: float,
) -> dict[str, torch.Tensor]:
    device = next(head.parameters()).device
    text_embeddings = batch["text_embeddings"].to(device, non_blocking=True)
    text_padding_mask = None
    if head.training and prompt_dropout > 0.0 and text_embeddings.ndim == 3:
        text_padding_mask = (
            torch.rand(text_embeddings.shape[:2], device=text_embeddings.device)
            < prompt_dropout
        )
        all_dropped = text_padding_mask.all(dim=1)
        if all_dropped.any():
            keep_indices = torch.randint(
                text_embeddings.shape[1],
                (int(all_dropped.sum().item()),),
                device=text_embeddings.device,
            )
            text_padding_mask[all_dropped, keep_indices] = False
    target_masks = batch["target_masks"].to(device, non_blocking=True)
    mask_output_size = target_masks.shape[-2:]
    object_image_indices = batch["object_image_indices"].to(device, non_blocking=True)
    object_boxes = batch["object_boxes"].to(device, non_blocking=True)
    object_boxes.clamp_(0.0, 1.0)
    object_queries = None

    if "multi_image_features" in batch:
        if head.query_dim is not None:
            raise RuntimeError(
                "Detector-query conditioning requires an online training stage; "
                "the current feature cache does not contain detector queries."
            )
        multi_image_features = [
            feature_level.to(device, non_blocking=True)
            for feature_level in batch["multi_image_features"]
        ]
    else:
        if online_encoder is None:
            raise RuntimeError("Online stage requires a frozen detector encoder.")
        with torch.no_grad():
            detector_output = extract_online_detector_output(
                online_encoder,
                batch["images"],
            )
        multi_image_features = [
            feature_level.to(device, non_blocking=True)
            for feature_level in detector_output.feature_maps
        ]
        if head.query_dim is not None:
            query_embeddings = detector_output.query_embeddings
            query_logits = detector_output.query_logits
            query_boxes = detector_output.query_boxes
            if not all(
                isinstance(value, torch.Tensor)
                for value in (query_embeddings, query_logits, query_boxes)
            ):
                raise RuntimeError(
                    "The selected detector does not expose decoder queries, logits, "
                    "and boxes required by this head."
                )
            typed_embeddings = cast(torch.Tensor, query_embeddings)
            typed_logits = cast(torch.Tensor, query_logits)
            typed_boxes = cast(torch.Tensor, query_boxes)
            if typed_embeddings.shape[-1] != head.query_dim:
                raise RuntimeError(
                    f"Detector query dimension {typed_embeddings.shape[-1]} does not "
                    f"match the configured head dimension {head.query_dim}."
                )
            object_labels = batch["object_labels"].to(device, non_blocking=True)
            matched_query_indices = match_object_queries(
                typed_logits,
                typed_boxes,
                object_image_indices,
                object_labels,
                object_boxes,
            )
            object_queries = typed_embeddings[
                object_image_indices,
                matched_query_indices,
            ]

    multi_image_features = [
        feature_level.index_select(0, object_image_indices)
        for feature_level in multi_image_features
    ]

    with build_autocast_context(enabled=use_amp, device=device):
        outputs = head(
            multi_image_features,
            text_embeddings,
            text_padding_mask=text_padding_mask,
            object_boxes=object_boxes,
            object_queries=object_queries,
            mask_output_size=mask_output_size,
            return_intermediates=False,
        )
        mask_logits = cast(torch.Tensor, outputs["mask_logits"])
        return compute_segmentation_objective(
            mask_logits,
            target_masks,
            object_image_indices,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            dice_smooth=dice_smooth,
        )


def build_optimizer(model: BaseFusionHead, *, lr: float, weight_decay: float) -> AdamW:
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def save_training_checkpoint(
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
    architecture_config = getattr(head, "architecture_config", None)
    if not isinstance(architecture_config, dict):
        raise TypeError("BaseFusionHead is missing its architecture configuration.")
    save_head_bundle(
        checkpoint_dir,
        config=architecture_config,
        state_dict=head.state_dict(),
    )
    save_training_state(
        checkpoint_dir,
        optimizer,
        scheduler,
        epoch=epoch,
        global_step=global_step,
        scaler=scaler,
    )
    atomic_write_text(
        output_dir / "last_checkpoint",
        checkpoint_dir.name + "\n",
    )
    return checkpoint_dir


def load_training_checkpoint(
    head: BaseFusionHead,
    checkpoint_dir: Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[int, int]:
    checkpoint = load_head_bundle(checkpoint_dir / "head.pt")
    architecture_config = getattr(head, "architecture_config", None)
    if checkpoint.config != architecture_config:
        raise ValueError(
            "The resume checkpoint head configuration does not match the current "
            "training configuration."
        )
    head.load_state_dict(checkpoint.state_dict)
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
    online_encoder: Detector | None,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    bce_weight: float,
    dice_weight: float,
    dice_smooth: float,
    prompt_dropout: float,
) -> tuple[dict[str, Any], int]:
    start = time.perf_counter()
    head.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    batch_count = 0
    first_loss: float | None = None
    last_loss: float | None = None

    for batch in data_loader:
        if max_steps is not None and global_step >= max_steps:
            break

        batch_result = compute_batch_loss(
            head,
            batch,
            online_encoder=online_encoder,
            use_amp=use_amp,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            dice_smooth=dice_smooth,
            prompt_dropout=prompt_dropout,
        )
        loss = batch_result["loss"]

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
    end = time.perf_counter()
    return (
        {
            "epoch": epoch,
            "phase": phase,
            "global_step": global_step,
            "loss": round(epoch_loss, 6),
            "first_loss": None if first_loss is None else round(first_loss, 6),
            "last_loss": None if last_loss is None else round(last_loss, 6),
            "time": end - start,
        },
        global_step,
    )


@torch.no_grad()
def evaluate(
    head: BaseFusionHead,
    data_loader: DataLoader,
    *,
    epoch: int,
    phase: PhaseName,
    online_encoder: Detector | None,
    use_amp: bool,
    bce_weight: float,
    dice_weight: float,
    dice_smooth: float,
) -> dict[str, Any]:
    head.eval()
    total_loss = 0.0
    batch_count = 0
    metric_totals = {name: 0.0 for name in ("bce", "dice_loss", "dice", "iou", "brier")}

    for batch in data_loader:
        batch_result = compute_batch_loss(
            head,
            batch,
            online_encoder=online_encoder,
            use_amp=use_amp,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            dice_smooth=dice_smooth,
            prompt_dropout=0.0,
        )
        loss = batch_result["loss"]
        total_loss += float(loss.item())
        for name in metric_totals:
            metric_totals[name] += float(batch_result[name].item())
        batch_count += 1

    average_loss = total_loss / max(batch_count, 1)
    metrics = {
        name: round(total / max(batch_count, 1), 6)
        for name, total in metric_totals.items()
    }
    summary = {
        "epoch": epoch,
        "phase": phase,
        "loss": round(average_loss, 6),
        **metrics,
    }
    log_event("validation_epoch", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.bce_loss_weight < 0 or args.dice_loss_weight < 0:
        raise ValueError("Loss weights must be non-negative.")
    if args.bce_loss_weight == 0 and args.dice_loss_weight == 0:
        raise ValueError("At least one segmentation loss weight must be positive.")
    if args.dice_smooth <= 0:
        raise ValueError("dice_smooth must be positive.")
    if not 0.0 <= args.prompt_dropout <= 1.0:
        raise ValueError("prompt_dropout must be between zero and one.")
    if not args.mask_upsample_scales or any(
        scale <= 0 for scale in args.mask_upsample_scales
    ):
        raise ValueError("mask_upsample_scales must contain positive integers.")
    if args.use_detector_query and args.detector_query_dim <= 0:
        raise ValueError("detector_query_dim must be positive.")
    set_seed(args.seed)

    device = torch.device(args.device) if args.device is not None else get_device()
    log_event("device", {"device": str(device)})
    dataset = dataset_from_args(args)
    ensure_split_exists(dataset, args.train_split, role="train")
    ensure_split_exists(dataset, args.validation_split, role="validation")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = output_dir / "training_log.jsonl" if args.save_logs else None

    stages = parse_stage_spec(args.train_stages)
    if args.use_detector_query and any(phase == "offline" for phase, _ in stages):
        raise ValueError(
            "Detector-query conditioning currently requires online-only training "
            "because cached features do not include detector queries."
        )
    train_loaders: dict[PhaseName, DataLoader] = {}
    validation_loaders: dict[PhaseName, DataLoader] = {}

    if any(phase == "online" for phase, _ in stages):
        online_train_dataset = build_online_dataset(dataset[args.train_split], args)
        online_validation_dataset = build_validation_dataset(
            dataset[args.validation_split], args
        )
        train_loaders["online"] = build_dataloader(
            online_train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            seed=args.seed,
            phase="online",
            image_size=args.image_size,
            expected_levels=args.num_feature_levels,
            expected_channels=args.image_dim,
            pin_memory=device.type == "cuda",
        )
        validation_loaders["online"] = build_dataloader(
            online_validation_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            seed=args.seed,
            phase="online",
            image_size=args.image_size,
            expected_levels=args.num_feature_levels,
            expected_channels=args.image_dim,
            pin_memory=device.type == "cuda",
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
            expected_levels=args.num_feature_levels,
            expected_channels=args.image_dim,
            pin_memory=device.type == "cuda",
        )
        validation_loaders["offline"] = build_dataloader(
            offline_validation_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            seed=args.seed,
            phase="offline",
            image_size=args.image_size,
            expected_levels=args.num_feature_levels,
            expected_channels=args.image_dim,
            pin_memory=device.type == "cuda",
        )

    head = build_head(args, device)
    online_encoder = (
        build_frozen_encoder(args, device) if "online" in train_loaders else None
    )
    run_config_path = write_run_manifest(
        output_dir,
        args,
        model_config=cast(dict[str, Any], head.architecture_config),
        resolved={
            "device": str(device),
            "output_dir": str(output_dir),
            "detector_architecture": (
                None if online_encoder is None else online_encoder.architecture
            ),
            "detector_model_source": (
                None if online_encoder is None else online_encoder.source
            ),
            "train_stages": [
                {"phase": phase, "epochs": epochs} for phase, epochs in stages
            ],
        },
    )
    log_event("saved_run_config", {"path": str(run_config_path)})
    optimizer = build_optimizer(head, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = estimate_total_steps(stages, train_loaders, max_steps=args.max_steps)
    scheduler = build_scheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps
    )
    scaler = build_grad_scaler(enabled=args.amp, device=device)

    start_epoch = 0
    global_step = 0
    resume_checkpoint = resolve_resume_checkpoint(args.resume_from)
    if resume_checkpoint is not None:
        start_epoch, global_step = load_training_checkpoint(
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
                online_encoder=online_encoder if phase == "online" else None,
                use_amp=args.amp,
                scaler=scaler,
                bce_weight=args.bce_loss_weight,
                dice_weight=args.dice_loss_weight,
                dice_smooth=args.dice_smooth,
                prompt_dropout=args.prompt_dropout,
            )
            log_event("train_epoch", epoch_summary)
            if training_log_path is not None:
                append_log_event(training_log_path, "train_epoch", epoch_summary)

            validation_summary = evaluate(
                head,
                validation_loader,
                epoch=epoch_index,
                phase=phase,
                online_encoder=online_encoder if phase == "online" else None,
                use_amp=args.amp,
                bce_weight=args.bce_loss_weight,
                dice_weight=args.dice_loss_weight,
                dice_smooth=args.dice_smooth,
            )
            if training_log_path is not None:
                append_log_event(
                    training_log_path,
                    "validation_epoch",
                    validation_summary,
                )

            if (epoch_index + 1) % args.save_every_epochs == 0:
                checkpoint_dir = save_training_checkpoint(
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
    architecture_config = getattr(head, "architecture_config", None)
    if not isinstance(architecture_config, dict):
        raise TypeError("BaseFusionHead is missing its architecture configuration.")
    final_path = save_head_bundle(
        final_dir,
        config=architecture_config,
        state_dict=head.state_dict(),
    )
    log_event("saved_final_model", {"path": str(final_path)})


if __name__ == "__main__":
    main()
