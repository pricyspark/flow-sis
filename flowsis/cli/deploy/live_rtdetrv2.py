import cv2
import time
import argparse
from typing import Any
from pathlib import Path
from numpy.typing import NDArray
from collections.abc import Mapping

from flowsis.utils import get_device
from flowsis.pretrained.rtdetrv2 import RTDetrV2


WINDOW_NAME = "RT-DETRv2 Live Inference"
BOX_COLORS = [
    (255, 90, 95),
    (46, 196, 182),
    (255, 191, 105),
    (56, 103, 214),
    (106, 76, 147),
    (32, 191, 85),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live inference with a trained RT-DETRv2 object detector.")
    parser.add_argument("--model_path", type=Path, default=Path("outputs/rtdetrv2/final"))
    parser.add_argument(
        "--video_source",
        type=str,
        default="live",
        help="Use 'live' for the default webcam, a camera index such as '1', or a video file path.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=640)
    return parser.parse_args()


def resolve_video_source(video_source: str) -> int | str:
    if video_source == "live":
        return 0
    if video_source.isdigit():
        return int(video_source)
    return video_source


def load_id2label(model: RTDetrV2) -> dict[int, str]:
    raw_id2label = getattr(model.model.config, "id2label", None)
    if isinstance(raw_id2label, Mapping):
        normalized = {
            int(label_id): str(label_name) if label_name else f"class_{int(label_id)}"
            for label_id, label_name in raw_id2label.items()
        }
        if normalized:
            return normalized

    num_labels = int(getattr(model.model.config, "num_labels", 0) or 0)
    return {label_id: f"class_{label_id}" for label_id in range(num_labels)}


def color_for_label(label_id: int) -> tuple[int, int, int]:
    return BOX_COLORS[label_id % len(BOX_COLORS)]


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
    text_padding = 6
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = origin
    x1 = max(0, x)
    y1 = max(0, y - text_height - baseline - text_padding * 2)
    x2 = min(frame.shape[1], x1 + text_width + text_padding * 2)
    y2 = min(frame.shape[0], y)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.putText(
        frame,
        text,
        (x1 + text_padding, y2 - baseline - text_padding),
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
    id2label: dict[int, str],
    inference_ms: float,
) -> Any:
    rendered = frame_bgr.copy()

    boxes = detections["boxes"].detach().cpu().tolist()
    scores = detections["scores"].detach().cpu().tolist()
    labels = detections["labels"].detach().cpu().tolist()

    for box, score, label_id in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        label_id = int(label_id)
        color = color_for_label(label_id)
        label_name = id2label.get(label_id, f"class_{label_id}")
        label_text = f"{label_name} {float(score):.2f}"

        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, thickness=2)
        text_origin_y = y1 if y1 > 28 else min(rendered.shape[0] - 1, y2 + 28)
        draw_text_block(rendered, text=label_text, origin=(x1, text_origin_y), color=color)

    header = f"detections={len(labels)}  inference={inference_ms:.1f}ms"
    draw_text_block(rendered, text=header, origin=(8, 32), color=(0, 0, 0))
    return rendered


def crop_center_square(arr: NDArray) -> NDArray:
    H, W = arr.shape[:2]
    size = min(H, W)
    
    top = (H - size) // 2
    left = (W - size) // 2
    
    return arr[top:top + size, left:left + size]


def main() -> None:
    args = parse_args()
    device = get_device()

    model = RTDetrV2.from_pretrained(str(args.model_path), device=device)
    id2label = load_id2label(model)

    video_source = resolve_video_source(args.video_source)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.video_source}")

    print(
        "live_rtdetrv2",
        {
            "model_path": str(args.model_path),
            "video_source": str(args.video_source),
            "threshold": args.threshold,
            "image_size": args.image_size,
            "labels": id2label,
        },
    )

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            square_rgb = crop_center_square(frame_rgb)
            square_bgr = crop_center_square(frame_bgr)
            start_time = time.perf_counter()
            print(frame_rgb.shape)
            inference = model.infer([square_rgb], image_size=args.image_size, threshold=args.threshold)
            inference_ms = (time.perf_counter() - start_time) * 1000.0

            rendered_frame = draw_detections(
                square_bgr,
                inference.detections[0],
                id2label=id2label,
                inference_ms=inference_ms,
            )
            cv2.imshow(WINDOW_NAME, rendered_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
