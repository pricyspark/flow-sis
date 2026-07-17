from __future__ import annotations

import argparse
import json
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from flowsis.base_head import BaseFusionHead
from flowsis.pretrained.rtdetrv2 import RTDetrV2
from flowsis.selection import SelectionResult, select_first_detection, select_recurrent_detection
from flowsis.utils import build_autocast_context, get_device


WINDOW_NAME = "FlowSIS Live Inference"
DEFAULT_HEAD_CONFIG: dict[str, Any] = {
    "num_decode_layers": 1,
    "decode_embed_dim": 128,
    "image_dim": 256,
    "text_dim": 768,
    "nhead": 8,
    "decode_ffn_dim": 512,
    "dropout": 0.1,
    "activation": "gelu",
    "num_feature_levels": 3,
    "decode_pos_encode": "first",
    "image_self_attention": "WINDOW",
    "decode_window_size": 8,
    "use_shifted_windows": True,
    "multiscale_merge": "conv",
    "conv_merge_refinement": "depthwise",
    "deformable_num_points": 4,
    "deformable_offset_scale": 2.0,
    "aggregator_dim": None,
    "channel_aggregation": "none",
    "mask_feature_source": "merged",
    "mask_head_hidden_dim": None,
    "mask_output_dim": 1,
    "mask_convolution": "depthwise_separable",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RT-DETRv2 selection and FlowSIS mask inference on a video or camera feed."
    )
    parser.add_argument("--detector_path", type=Path, default=Path("outputs/rtdetrv2/final"))
    parser.add_argument("--head_path", type=Path, default=Path("outputs/base/final"))
    parser.add_argument(
        "--text_embeddings_dir",
        type=Path,
        default=Path("data/manifests/text-embeddings"),
    )
    parser.add_argument(
        "--video_source",
        default="live",
        help="Use 'live', a camera index such as '1', or a video file path.",
    )
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--detection_threshold", type=float, default=0.5)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--mask_alpha", type=float, default=0.45)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--history_size", type=int, default=12)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--cpu_preprocess",
        action="store_true",
        help="Resize, rescale, and normalize detector inputs on CPU instead of CUDA.",
    )
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    return parser.parse_args()


def resolve_video_source(source: str) -> int | str:
    if source == "live":
        return 0
    if source.isdigit():
        return int(source)
    return source


def resolve_head_weights(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [path / "model.pt", path / "final" / "model.pt"]
    last_checkpoint = path / "last_checkpoint"
    if last_checkpoint.exists():
        checkpoint = Path(last_checkpoint.read_text().strip())
        if not checkpoint.is_absolute():
            checkpoint = path / checkpoint
        candidates.append(checkpoint / "model.pt")
    candidates.extend(
        checkpoint / "model.pt"
        for checkpoint in sorted(path.glob("checkpoint-*"), reverse=True)
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find base-head model.pt under {path}.")


def load_head(
    head_path: Path,
    *,
    device: torch.device,
) -> tuple[BaseFusionHead, Path]:
    weights_path = resolve_head_weights(head_path)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    resolved_config = weights_path.with_name("head_config.json")
    if resolved_config.exists():
        config = json.loads(resolved_config.read_text())
    else:
        config = infer_head_config(state_dict)
        print(
            "head_config_warning",
            {
                "message": (
                    "No head_config.json found; inferred shape-dependent settings from weights "
                    "and used training defaults for settings not encoded in the state dict."
                ),
                "weights": str(weights_path),
                "inferred_config": config,
            },
        )
    if not isinstance(config, dict):
        raise TypeError(f"Expected a JSON object in {resolved_config}.")

    head = BaseFusionHead(**config).to(device)
    head.load_state_dict(state_dict)
    head.eval()
    return head, weights_path


def infer_head_config(state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Recover architecture values represented by legacy checkpoint tensor shapes."""
    config = dict(DEFAULT_HEAD_CONFIG)

    block_indices = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("decoder.blocks.") and key.split(".")[2].isdigit()
    }
    config["num_decode_layers"] = max(block_indices, default=-1) + 1

    level_embedding = state_dict.get("decoder.level_embedding.weight")
    if level_embedding is not None:
        config["num_feature_levels"] = int(level_embedding.shape[0])
        config["decode_embed_dim"] = int(level_embedding.shape[1])

    input_projection = state_dict.get("decoder.input_projections.0.weight")
    config["image_dim"] = (
        int(input_projection.shape[1])
        if input_projection is not None
        else int(config["decode_embed_dim"])
    )
    text_norm = state_dict.get("decoder.blocks.0.text_norm.weight")
    if text_norm is not None:
        config["text_dim"] = int(text_norm.shape[0])
    ffn_weight = state_dict.get("decoder.blocks.0.ffn.0.weight")
    if ffn_weight is not None:
        config["decode_ffn_dim"] = int(ffn_weight.shape[0])

    if any(key.startswith("decoder.deformable_fuse.") for key in state_dict):
        config["multiscale_merge"] = "deformable"
    elif any(key.startswith("decoder.level_fuse.") for key in state_dict):
        config["multiscale_merge"] = "conv"
        refinement = state_dict.get("decoder.level_fuse.2.weight")
        if refinement is not None:
            config["conv_merge_refinement"] = (
                "depthwise" if refinement.shape[1] == 1 else "standard"
            )
    else:
        config["multiscale_merge"] = "none"
        config["mask_feature_source"] = "highest_resolution"

    aggregator_weight = state_dict.get("channel_aggregator.image_proj.weight")
    if aggregator_weight is None:
        config["channel_aggregation"] = "none"
        config["aggregator_dim"] = None
    else:
        config["aggregator_dim"] = int(aggregator_weight.shape[0])
        # Sigmoid and softmax have identical parameter shapes. Sigmoid is the
        # historical constructor default and is the safest legacy fallback.
        config["channel_aggregation"] = "sigmoid"

    mask_projection = state_dict.get("mask_head.input_proj.0.weight")
    if mask_projection is not None:
        config["mask_head_hidden_dim"] = int(mask_projection.shape[0])
    logit_weight = state_dict.get("mask_head.logit_head.weight")
    if logit_weight is not None:
        config["mask_output_dim"] = int(logit_weight.shape[0])
    config["mask_convolution"] = (
        "depthwise_separable"
        if any(key.startswith("mask_head.blocks.0.block.0.0.") for key in state_dict)
        else "standard"
    )
    return config


def load_id2label(model: RTDetrV2) -> dict[int, str]:
    raw = getattr(model.model.config, "id2label", None)
    if isinstance(raw, Mapping):
        labels = {int(index): str(label) for index, label in raw.items()}
        if labels:
            return labels
    num_labels = int(getattr(model.model.config, "num_labels", 0) or 0)
    return {index: f"class_{index}" for index in range(num_labels)}


def load_prompt_embeddings(
    label: str,
    directory: Path,
    cache: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    if label not in cache:
        path = directory / f"{label}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing prompt embeddings for detector label {label!r}: {path}")
        embeddings = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
            shape = getattr(embeddings, "shape", None)
            raise ValueError(f"Expected prompt embeddings shaped [P,D] at {path}, got {shape}.")
        cache[label] = embeddings.float().to(device)
    return cache[label]


def center_square(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    size = min(height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return frame[top : top + size, left : left + size]


def preprocess_on_gpu(
    frame_bgr: np.ndarray,
    *,
    image_size: int,
    model: RTDetrV2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the detector image processor's numerical transforms on CUDA."""
    processor = model.processor
    pixels = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(model.device)
    pixels = pixels.permute(2, 0, 1).flip(0).unsqueeze(0).float()
    pixels = F.interpolate(
        pixels,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    pixels.round_().clamp_(0.0, 255.0)

    if getattr(processor, "do_rescale", True):
        pixels.mul_(float(getattr(processor, "rescale_factor", 1.0 / 255.0)))
    if getattr(processor, "do_normalize", False):
        mean = torch.as_tensor(processor.image_mean, device=model.device).view(1, 3, 1, 1)
        std = torch.as_tensor(processor.image_std, device=model.device).view(1, 3, 1, 1)
        pixels.sub_(mean).div_(std)

    pixel_mask = torch.ones(
        (1, image_size, image_size),
        dtype=torch.bool,
        device=model.device,
    )
    return pixels, pixel_mask


def select_detection(
    history: deque[Mapping[str, Any]],
    previous: SelectionResult | None,
) -> SelectionResult | None:
    if not history or len(history[-1]["scores"]) == 0:
        return None
    if previous is None:
        return select_first_detection(history)
    return select_recurrent_detection(history, previous)


def normalized_box(selection: SelectionResult, *, width: int, height: int) -> torch.Tensor:
    x1, y1, x2, y2 = selection.box
    return torch.tensor(
        [[x1 / width, y1 / height, x2 / width, y2 / height]],
        dtype=torch.float32,
    ).clamp_(0.0, 1.0)


@torch.inference_mode()
def predict_mask(
    head: BaseFusionHead,
    feature_maps: list[torch.Tensor],
    text_embeddings: torch.Tensor,
    box: torch.Tensor,
    *,
    output_size: tuple[int, int],
    device: torch.device,
    use_amp: bool,
    mask_threshold: float,
) -> np.ndarray:
    with build_autocast_context(enabled=use_amp, device=device):
        output = head(
            feature_maps,
            text_embeddings.unsqueeze(0),
            object_boxes=box.to(device),
            mask_output_size=output_size,
            return_intermediates=False,
        )
    # Threshold on the GPU so only one byte per pixel crosses PCIe instead of
    # a four-byte probability map. This produces the same binary mask used by
    # rendering and avoids CPU-side sigmoid and threshold work.
    mask = cast(torch.Tensor, output["mask_logits"])[0].sigmoid() >= mask_threshold
    return mask.to(dtype=torch.uint8).cpu().numpy()


def render_result(
    frame: np.ndarray,
    mask: np.ndarray | None,
    selection: SelectionResult | None,
    *,
    label: str | None,
    mask_alpha: float,
    elapsed_ms: float,
) -> np.ndarray:
    rendered = frame.copy()
    if mask is not None:
        if cv2.countNonZero(mask):
            overlay = np.empty_like(rendered)
            overlay[:] = (0, 220, 0)
            blended = cv2.addWeighted(rendered, 1.0 - mask_alpha, overlay, mask_alpha, 0.0)
            cv2.copyTo(blended, mask, rendered)

    if selection is not None:
        x1, y1, x2, y2 = (int(round(value)) for value in selection.box)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 255, 255), 2)
        text = f"{label or selection.label} det={selection.score:.2f} sel={selection.selection_score:.2f}"
        cv2.putText(
            rendered,
            text,
            (max(x1, 0), max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        rendered,
        f"{elapsed_ms:.1f} ms | q/esc: quit",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return rendered


def open_writer(path: Path, *, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output video: {path}")
    return writer


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.detection_threshold <= 1.0:
        raise ValueError("--detection_threshold must be between zero and one.")
    if not 0.0 <= args.mask_threshold <= 1.0:
        raise ValueError("--mask_threshold must be between zero and one.")
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask_alpha must be between zero and one.")
    if args.history_size <= 0:
        raise ValueError("--history_size must be positive.")
    if args.no_display and args.output_path is None:
        raise ValueError("--no_display requires --output_path.")

    device = torch.device(args.device) if args.device else get_device()
    detector = RTDetrV2.from_pretrained(str(args.detector_path), device=device)
    detector.eval()
    head, weights_path = load_head(args.head_path, device=device)
    id2label = load_id2label(detector)
    prompt_cache: dict[str, torch.Tensor] = {}
    history: deque[Mapping[str, Any]] = deque(maxlen=args.history_size)
    previous_selection: SelectionResult | None = None

    capture = cv2.VideoCapture(resolve_video_source(args.video_source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.video_source}")

    writer: cv2.VideoWriter | None = None
    frame_count = 0
    print(
        "live_base",
        {
            "detector": str(args.detector_path),
            "head": str(weights_path),
            "video_source": args.video_source,
            "device": str(device),
        },
    )

    try:
        while args.max_frames is None or frame_count < args.max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            square_bgr = center_square(frame_bgr)
            height, width = square_bgr.shape[:2]
            start = time.perf_counter()

            if args.cpu_preprocess:
                square_rgb = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB)
                inference = detector.infer(
                    [square_rgb],
                    image_size=args.image_size,
                    threshold=args.detection_threshold,
                )
            else:
                pixel_values, pixel_mask = preprocess_on_gpu(
                    square_bgr,
                    image_size=args.image_size,
                    model=detector,
                )
                inference = detector.infer_preprocessed(
                    pixel_values,
                    pixel_mask=pixel_mask,
                    original_sizes=[(height, width)],
                    threshold=args.detection_threshold,
                )
            gpu_detections = inference.detections[0]
            # Selection revisits every history entry for each candidate. Copy
            # detections once instead of synchronizing CUDA repeatedly there.
            detections = {
                key: value.detach().cpu().tolist()
                for key, value in gpu_detections.items()
            }
            history.append(detections)
            selection = select_detection(history, previous_selection)
            mask: np.ndarray | None = None
            selected_label: str | None = None
            if selection is not None:
                selected_label = id2label.get(selection.label, f"class_{selection.label}")
                text_embeddings = load_prompt_embeddings(
                    selected_label,
                    args.text_embeddings_dir,
                    prompt_cache,
                    device=device,
                )
                feature_maps = [feature.float() for feature in inference.encodings]
                mask = predict_mask(
                    head,
                    feature_maps,
                    text_embeddings,
                    normalized_box(selection, width=width, height=height),
                    output_size=(height, width),
                    device=device,
                    use_amp=args.amp,
                    mask_threshold=args.mask_threshold,
                )
                previous_selection = selection

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            rendered = render_result(
                square_bgr,
                mask,
                selection,
                label=selected_label,
                mask_alpha=args.mask_alpha,
                elapsed_ms=elapsed_ms,
            )

            if args.output_path is not None:
                if writer is None:
                    writer = open_writer(
                        args.output_path,
                        fps=float(capture.get(cv2.CAP_PROP_FPS)),
                        size=(rendered.shape[1], rendered.shape[0]),
                    )
                writer.write(rendered)
            if not args.no_display:
                cv2.imshow(WINDOW_NAME, rendered)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    break
            frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print("live_base_complete", {"frames": frame_count, "output_path": str(args.output_path)})


if __name__ == "__main__":
    main()
