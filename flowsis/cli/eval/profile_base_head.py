import argparse
import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Literal, cast

import torch

from flowsis.base_head import BaseFusionHead
from flowsis.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile FlowSIS base-head components."
    )
    parser.add_argument("--preset", choices=("speed", "original"), default="speed")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--feature-size", type=int, default=80)
    parser.add_argument("--output-size", type=int, default=640)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace-path", type=Path, default=None)
    return parser.parse_args()


def build_head(preset: str, device: torch.device) -> BaseFusionHead:
    speed = preset == "speed"
    return (
        BaseFusionHead(
            num_decode_layers=1 if speed else 2,
            decode_embed_dim=128 if speed else 256,
            image_dim=256,
            text_dim=768,
            nhead=8,
            decode_ffn_dim=512 if speed else 1024,
            dropout=0.1,
            activation="gelu",
            num_feature_levels=3,
            decode_pos_encode="first",
            image_self_attention="WINDOW",
            decode_window_size=8,
            use_shifted_windows=True,
            multiscale_merge="conv",
            deformable_num_points=4,
            deformable_offset_scale=2.0,
            conv_merge_refinement="depthwise" if speed else "standard",
            channel_aggregation="none" if speed else "sigmoid",
            mask_feature_source="merged",
            mask_output_dim=1,
            mask_convolution="depthwise_separable" if speed else "standard",
        )
        .eval()
        .to(device)
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    operation: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    autocast_context: Callable[[], object],
) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            with autocast_context():
                operation()
        synchronize(device)

        samples = []
        for _ in range(iterations):
            synchronize(device)
            start = time.perf_counter()
            with autocast_context():
                operation()
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(samples)
    p95_index = min(round(0.95 * (len(ordered) - 1)), len(ordered) - 1)
    median = statistics.median(samples)
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": median,
        "p95_ms": ordered[p95_index],
        "fps_from_median": 1000.0 / median,
    }


def main() -> None:
    args = parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup must be non-negative.")
    device = torch.device(args.device) if args.device is not None else get_device()
    use_amp = bool(args.amp and device.type == "cuda")

    def autocast_context():
        if not use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    head = build_head(args.preset, device)
    size = args.feature_size
    image_features = [
        torch.randn(args.batch_size, 256, size // scale, size // scale, device=device)
        for scale in (1, 2, 4)
    ]
    text_embeddings = torch.randn(
        args.batch_size,
        args.num_prompts,
        768,
        device=device,
    )

    with torch.inference_mode(), autocast_context():
        fused = cast(
            list[torch.Tensor],
            head.decoder(image_features, text_embeddings, return_merged_features=False),
        )
        merged = head.decoder._get_merged_features(fused, text_embeddings)
        if merged is None:
            raise RuntimeError("Profiling requires a merged mask feature.")
        modulated, _, _ = head._apply_channel_aggregation(merged, text_embeddings)

    operations: dict[str, Callable[[], object]] = {
        "fusion_levels": lambda: head.decoder(
            image_features, text_embeddings, return_merged_features=False
        ),
        "multiscale_merge": lambda: head.decoder._get_merged_features(
            fused, text_embeddings
        ),
        "channel_conditioning": lambda: head._apply_channel_aggregation(
            merged, text_embeddings
        ),
        "mask_decoder": lambda: head.mask_head(
            modulated,
            text_embeddings,
            output_size=(args.output_size, args.output_size),
        ),
        "full_head": lambda: head(
            image_features,
            text_embeddings,
            mask_output_size=(args.output_size, args.output_size),
            return_intermediates=False,
        ),
    }

    results = {
        name: benchmark(
            operation,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            autocast_context=autocast_context,
        )
        for name, operation in operations.items()
    }

    peak_memory_mb = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), autocast_context():
            operations["full_head"]()
        synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    if args.trace_path is not None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        args.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as profile:
            with torch.inference_mode(), autocast_context():
                operations["full_head"]()
        profile.export_chrome_trace(str(args.trace_path))

    summary = {
        "preset": args.preset,
        "device": str(device),
        "amp": use_amp,
        "batch_size": args.batch_size,
        "feature_shapes": [list(feature.shape) for feature in image_features],
        "parameters": sum(parameter.numel() for parameter in head.parameters()),
        "peak_memory_mb": peak_memory_mb,
        "timings": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
