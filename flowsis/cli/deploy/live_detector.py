from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from typing import Any

import cv2

from flowsis.cli.common import add_detector_arguments, log_event
from flowsis.cli.deploy.common import center_square, resolve_video_source
from flowsis.pretrained import load_detector
from flowsis.utils import get_device


WINDOW_NAME = "FlowSIS Detector Live Inference"
BOX_COLORS = [
    (255, 90, 95),
    (46, 196, 182),
    (255, 191, 105),
    (56, 103, 214),
    (106, 76, 147),
    (32, 191, 85),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live inference with a supported detector."
    )
    add_detector_arguments(parser)
    parser.add_argument(
        "--video_source",
        default="live",
        help="Use 'live', a camera index such as '1', or a video file path.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--cpu_preprocess",
        action="store_true",
        help="Use the reference CPU image processor instead of device preprocessing.",
    )
    return parser.parse_args()


def draw_text_block(
    frame: Any,
    *,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    padding = 6
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    x, y = origin
    x1 = max(0, x)
    y1 = max(0, y - text_height - baseline - padding * 2)
    x2 = min(frame.shape[1], x1 + text_width + padding * 2)
    y2 = min(frame.shape[0], y)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.putText(
        frame,
        text,
        (x1 + padding, y2 - baseline - padding),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(
    frame_bgr: Any,
    detections: Mapping[str, Any],
    *,
    label_names: dict[int, str],
    inference_ms: float,
) -> Any:
    rendered = frame_bgr.copy()
    cpu = {
        key: value.detach().cpu().tolist() for key, value in detections.items()
    }
    for box, score, label_id in zip(
        cpu["boxes"],
        cpu["scores"],
        cpu["labels"],
    ):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        label_id = int(label_id)
        color = BOX_COLORS[label_id % len(BOX_COLORS)]
        label = label_names.get(label_id, f"class_{label_id}")
        text_y = y1 if y1 > 28 else min(rendered.shape[0] - 1, y2 + 28)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, thickness=2)
        draw_text_block(
            rendered,
            text=f"{label} {float(score):.2f}",
            origin=(x1, text_y),
            color=color,
        )
    draw_text_block(
        rendered,
        text=f"detections={len(cpu['labels'])}  inference={inference_ms:.1f}ms",
        origin=(8, 32),
        color=(0, 0, 0),
    )
    return rendered


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between zero and one.")
    device = get_device() if args.device is None else args.device
    detector = load_detector(
        args.model_name_or_path,
        architecture=args.detector_architecture,
        device=device,
    )
    detector.eval()
    log_event(
        "live_detector",
        {
            "architecture": detector.architecture,
            "model": detector.source,
            "video_source": args.video_source,
            "threshold": args.threshold,
            "image_size": args.image_size,
            "device": str(detector.device),
            "labels": detector.label_names,
        },
    )

    capture = cv2.VideoCapture(resolve_video_source(args.video_source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.video_source}")
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_bgr = center_square(frame_bgr)
            start = time.perf_counter()
            result = detector.infer_frame(
                frame_bgr,
                image_size=args.image_size,
                threshold=args.threshold,
                device_preprocess=not args.cpu_preprocess,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            cv2.imshow(
                WINDOW_NAME,
                draw_detections(
                    frame_bgr,
                    result.detections[0],
                    label_names=detector.label_names,
                    inference_ms=elapsed_ms,
                ),
            )
            if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
