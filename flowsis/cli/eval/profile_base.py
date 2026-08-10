from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.profiler import ProfilerActivity, profile, record_function

from flowsis.artifacts import atomic_write_text
from flowsis.base import FlowSISBase
from flowsis.cli.common import add_detector_arguments
from flowsis.cli.deploy.common import center_square, resolve_video_source
from flowsis.data import LabelPrompts
from flowsis.pretrained import BaseDetector
from flowsis.selection import normalize_box, select_detection
from flowsis.utils import build_autocast_context

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the complete FlowSIS base inference pipeline on video frames."
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
        required=True,
        help="Video file or camera source. Frames are loaded before timed inference.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--history-size", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CUDA FP16 autocast.",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly enable or disable TF32. By default, preserve PyTorch settings.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default=None,
        help="Set torch float32 matmul precision before loading the model.",
    )
    parser.add_argument(
        "--compile",
        choices=("none", "detector", "head", "both"),
        default="none",
        help="Compile the detector, fusion head, both, or neither.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="reduce-overhead",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Optionally write a Chrome/Perfetto trace for several inference steps.",
    )
    parser.add_argument("--trace-iterations", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one.")
    ordered = sorted(samples)
    index = round(quantile * (len(ordered) - 1))
    return ordered[index]


def summarize_timings(samples: Sequence[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("Cannot summarize an empty sequence of timings.")
    median = statistics.median(samples)
    return {
        "count": len(samples),
        "mean_ms": round(statistics.mean(samples), 4),
        "median_ms": round(median, 4),
        "p90_ms": round(percentile(samples, 0.90), 4),
        "p95_ms": round(percentile(samples, 0.95), 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
        "fps_from_median": round(1000.0 / median, 3),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(operation: Callable[[], T], *, device: torch.device) -> tuple[T, float]:
    synchronize(device)
    start = time.perf_counter()
    result = operation()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


def load_frames(source: str, *, count: int) -> list[NDArray[np.uint8]]:
    if count <= 0:
        raise ValueError("Frame count must be positive.")
    capture = cv2.VideoCapture(resolve_video_source(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    frames: list[NDArray[np.uint8]] = []
    try:
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(center_square(frame))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from video source: {source}")
    return frames


def configure_precision(args: argparse.Namespace) -> None:
    if args.matmul_precision is not None:
        torch.set_float32_matmul_precision(args.matmul_precision)
    if args.tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32


def compile_model(model: FlowSISBase, args: argparse.Namespace) -> None:
    compile_kwargs = {"mode": args.compile_mode, "dynamic": False}
    model._profile_has_compiled_modules = args.compile != "none"
    if args.compile in {"detector", "both"}:
        detector = cast(BaseDetector, model.detector)
        detector._model = torch.compile(
            detector._model,
            **compile_kwargs,
        )
    if args.compile in {"head", "both"}:
        model.head = torch.compile(model.head, **compile_kwargs)


def mark_inference_step(model: FlowSISBase) -> None:
    """Mark the lifetime boundary for outputs from compiled CUDA graphs."""
    if (
        model.device.type == "cuda"
        and getattr(model, "_profile_has_compiled_modules", False)
    ):
        torch.compiler.cudagraph_mark_step_begin()


def transfer_mask(mask: torch.Tensor | None) -> NDArray[np.uint8] | None:
    if mask is None:
        return None
    return mask.to(dtype=torch.uint8).cpu().numpy()


def run_end_to_end(
    model: FlowSISBase,
    frame: NDArray[np.uint8],
) -> NDArray[np.uint8] | None:
    mark_inference_step(model)
    return transfer_mask(model.infer(frame))


@torch.inference_mode()
def run_staged_frame(
    model: FlowSISBase,
    frame: NDArray[np.uint8],
) -> dict[str, float]:
    """Run the same base pipeline with synchronization at component boundaries."""
    device = model.device
    mark_inference_step(model)
    detector = cast(BaseDetector, model.detector)
    height, width = frame.shape[:2]
    timings: dict[str, float] = {}

    (pixels, pixel_mask), timings["preprocess"] = timed_call(
        lambda: detector.preprocess_bgr_frame(frame),
        device=device,
    )

    def detector_forward() -> Any:
        with build_autocast_context(enabled=model.use_amp, device=device):
            return detector._inference_model(
                pixels,
                pixel_mask,
            )

    outputs, timings["detector_forward"] = timed_call(
        detector_forward,
        device=device,
    )

    def detector_postprocess() -> Any:
        with build_autocast_context(enabled=model.use_amp, device=device):
            return detector._postprocess(
                outputs,
                original_sizes=[(height, width)],
                threshold=model.detection_threshold,
            )

    inference, timings["detector_postprocess"] = timed_call(
        detector_postprocess,
        device=device,
    )

    def select() -> Any:
        detections = {
            key: value.detach().cpu().tolist()
            for key, value in inference.detections[0].items()
        }
        model.history.append(detections)
        selection = select_detection(model.history, model.previous_selection)
        model.current_selection = selection
        return selection

    selection, timings["selection"] = timed_call(select, device=device)
    if selection is None:
        return timings

    def prepare_query() -> tuple[torch.Tensor, torch.Tensor]:
        selected_label = cast(str, model.selected_label)
        text_embeddings = LabelPrompts.load_embeddings(
            selected_label,
            model.text_embeddings_dir,
            model.prompt_cache,
            device=device,
        )
        box = torch.tensor(
            [normalize_box(selection, width=width, height=height)],
            device=device,
            dtype=torch.float32,
        )
        return text_embeddings, box

    (text_embeddings, box), timings["query_preparation"] = timed_call(
        prepare_query,
        device=device,
    )
    logits, timings["base_head"] = timed_call(
        lambda: model.predict_logits(
            list(inference.feature_maps),
            text_embeddings,
            box,
            output_size=(height, width),
        ),
        device=device,
    )
    model.previous_selection = selection
    mask = model.binarize_logits(logits)
    _, timings["mask_transfer"] = timed_call(
        lambda: transfer_mask(mask),
        device=device,
    )
    return timings


def benchmark_end_to_end(
    model: FlowSISBase,
    frames: Sequence[NDArray[np.uint8]],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    model.reset()
    for index in range(warmup):
        run_end_to_end(model, frames[index % len(frames)])
    synchronize(model.device)

    samples = []
    for index in range(iterations):
        frame = frames[(warmup + index) % len(frames)]
        _, elapsed = timed_call(
            lambda frame=frame: run_end_to_end(model, frame),
            device=model.device,
        )
        samples.append(elapsed)
    return summarize_timings(samples)


def benchmark_stages(
    model: FlowSISBase,
    frames: Sequence[NDArray[np.uint8]],
    *,
    iterations: int,
) -> dict[str, dict[str, float | int]]:
    model.reset()
    samples: defaultdict[str, list[float]] = defaultdict(list)
    for index in range(iterations):
        timings = run_staged_frame(model, frames[index % len(frames)])
        for name, elapsed in timings.items():
            samples[name].append(elapsed)
    return {name: summarize_timings(values) for name, values in samples.items()}


def write_trace(
    model: FlowSISBase,
    frames: Sequence[NDArray[np.uint8]],
    path: Path,
    *,
    iterations: int,
) -> None:
    if iterations <= 0:
        raise ValueError("Trace iterations must be positive.")
    activities = [ProfilerActivity.CPU]
    if model.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    model.reset()
    synchronize(model.device)
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for index in range(iterations):
            with record_function("flowsis_base_inference"):
                run_end_to_end(model, frames[index % len(frames)])
            profiler.step()
    synchronize(model.device)
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(path))
    print(
        profiler.key_averages().table(
            sort_by=(
                "self_cuda_time_total"
                if model.device.type == "cuda"
                else "self_cpu_time_total"
            ),
            row_limit=40,
        )
    )


def device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if device.type == "cuda":
        metadata.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
                "peak_memory_mb": round(
                    torch.cuda.max_memory_allocated(device) / (1024**2), 2
                ),
            }
        )
    return metadata


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("Warmup must be non-negative and iterations must be positive.")
    if not 0.0 <= args.detection_threshold <= 1.0:
        raise ValueError("Detection threshold must be between zero and one.")
    if not 0.0 <= args.mask_threshold <= 1.0:
        raise ValueError("Mask threshold must be between zero and one.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling was requested, but CUDA is unavailable.")
    configure_precision(args)

    frames = load_frames(
        args.video_source,
        count=max(args.warmup + args.iterations, args.trace_iterations),
    )
    model = FlowSISBase(
        args.detector_model_source,
        args.head_path,
        args.text_embeddings_dir,
        detector_architecture=args.detector_architecture,
        detection_threshold=args.detection_threshold,
        mask_threshold=args.mask_threshold,
        history_size=args.history_size,
        image_size=args.image_size,
        device=device,
        use_amp=args.amp,
        device_preprocess=True,
    )
    compile_model(model, args)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    end_to_end = benchmark_end_to_end(
        model,
        frames,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    stages = benchmark_stages(model, frames, iterations=args.iterations)
    metadata = device_metadata(device)

    if args.trace_path is not None:
        write_trace(
            model,
            frames,
            args.trace_path,
            iterations=args.trace_iterations,
        )

    summary = {
        "configuration": {
            "detector_architecture": model.detector.architecture,
            "detector": model.detector.source,
            "head": str(model.head_checkpoint_path),
            "video_source": args.video_source,
            "frames_loaded": len(frames),
            "image_size": args.image_size,
            "amp": args.amp,
            "compile": args.compile,
            "compile_mode": args.compile_mode,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "device": metadata,
        "end_to_end": end_to_end,
        "stages": stages,
        "notes": [
            "End-to-end timings reflect normal inference and should be used for latency.",
            "Stage timings synchronize at every boundary and are diagnostic; their sum may differ from end-to-end latency.",
            "Video decoding, rendering, and encoding are excluded because frames are preloaded.",
        ],
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        atomic_write_text(args.output_json, rendered + "\n")


if __name__ == "__main__":
    main()
