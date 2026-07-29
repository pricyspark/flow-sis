from __future__ import annotations

import argparse
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch

from flowsis.base_head import BaseFusionHead
from flowsis.cli.common import add_detector_arguments
from flowsis.cli.deploy.common import center_square, resolve_video_source
from flowsis.head_checkpoint import load_head_checkpoint, resolve_head_checkpoint
from flowsis.pretrained import load_detector
from flowsis.selection import SelectionResult, select_first_detection, select_recurrent_detection
from flowsis.utils import build_autocast_context, get_device


WINDOW_NAME = "FlowSIS Live Inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run detector selection and FlowSIS mask inference on a video or camera feed."
    )
    add_detector_arguments(
        parser,
        model_flag="--detector-model",
        model_dest="detector_model_source",
    )
    parser.add_argument("--head-path", type=Path, default=Path("outputs/base/final"))
    parser.add_argument(
        "--text-embeddings-dir",
        type=Path,
        default=Path("data/manifests/text-embeddings"),
    )
    parser.add_argument(
        "--video-source",
        default="live",
        help="Use 'live', a camera index such as '1', or a video file path.",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--history-size", type=int, default=12)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--cpu-preprocess",
        action="store_true",
        help="Resize, rescale, and normalize detector inputs on CPU instead of CUDA.",
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def load_head(
    head_path: Path,
    *,
    device: torch.device,
) -> tuple[BaseFusionHead, Path]:
    checkpoint_path = resolve_head_checkpoint(head_path)
    checkpoint = load_head_checkpoint(checkpoint_path)
    head = BaseFusionHead(**checkpoint.config).to(device)
    head.load_state_dict(checkpoint.state_dict)
    head.eval()
    return head, checkpoint_path


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
        raise ValueError("--detection-threshold must be between zero and one.")
    if not 0.0 <= args.mask_threshold <= 1.0:
        raise ValueError("--mask-threshold must be between zero and one.")
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be between zero and one.")
    if args.history_size <= 0:
        raise ValueError("--history-size must be positive.")
    if args.no_display and args.output_path is None:
        raise ValueError("--no-display requires --output-path.")

    device = torch.device(args.device) if args.device else get_device()
    detector = load_detector(
        args.detector_model_source,
        architecture=args.detector_architecture,
        image_size=args.image_size,
        device=device,
    )
    detector.eval()
    head, weights_path = load_head(args.head_path, device=device)
    id2label = detector.label_names
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
            "detector_architecture": detector.architecture,
            "detector": detector.source,
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
            inference = detector.infer_frame(
                square_bgr,
                threshold=args.detection_threshold,
                device_preprocess=not args.cpu_preprocess,
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
                feature_maps = [
                    feature.float() for feature in inference.feature_maps
                ]
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
