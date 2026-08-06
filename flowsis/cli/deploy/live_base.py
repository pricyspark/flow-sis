from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from flowsis.base import FlowSISBase
from flowsis.cli.common import add_detector_arguments
from flowsis.cli.deploy.common import center_square, resolve_video_source
from flowsis.selection import SelectionResult

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
            blended = cv2.addWeighted(
                rendered, 1.0 - mask_alpha, overlay, mask_alpha, 0.0
            )
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

    model = FlowSISBase(
        args.detector_model_source,
        args.head_path,
        args.text_embeddings_dir,
        detector_architecture=args.detector_architecture,
        detection_threshold=args.detection_threshold,
        mask_threshold=args.mask_threshold,
        history_size=args.history_size,
        image_size=args.image_size,
        device=args.device,
        use_amp=args.amp,
        device_preprocess=not args.cpu_preprocess,
    )

    capture = cv2.VideoCapture(resolve_video_source(args.video_source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.video_source}")

    writer: cv2.VideoWriter | None = None
    frame_count = 0
    print(
        "live_base",
        {
            "detector_architecture": model.detector.architecture,
            "detector": model.detector.source,
            "head": str(model.head_checkpoint_path),
            "video_source": args.video_source,
            "device": str(model.device),
        },
    )

    try:
        while args.max_frames is None or frame_count < args.max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            square_bgr = center_square(frame_bgr)
            start = time.perf_counter()
            device_mask = model.infer(square_bgr)
            mask = (
                None
                if device_mask is None
                else device_mask.to(dtype=torch.uint8).cpu().numpy()
            )

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            rendered = render_result(
                square_bgr,
                mask,
                model.current_selection,
                label=model.selected_label,
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

    print(
        "live_base_complete",
        {"frames": frame_count, "output_path": str(args.output_path)},
    )


if __name__ == "__main__":
    main()
