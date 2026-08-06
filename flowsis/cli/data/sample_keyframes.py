import cv2
import time
import argparse
import csv
import imagehash
import numpy as np
from PIL import Image
from pathlib import Path
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Literal
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

DEFAULT_VIDEO_DIR = Path("data/raw")
DEFAULT_FRAME_DIR = Path("data/frames")
DEFAULT_MANIFEST_DIR = Path("data/manifests")
FRAME_MANIFEST_FIELDS = [
    "video_path",
    "id",
    "height",
    "width",
    "video_id",
    "cluster_id",
    "frame_idx",
    "blur_score",
    "cluster_size",
    "candidate_count",
    "avg_hamming",
    "output_path",
]


@dataclass
class FrameRecord:
    video_path: Path
    video_idx: int
    frame_idx: int
    phash: int
    blur_score: float
    frame_bgr: NDArray
    height: int
    width: int


@dataclass
class ClusterSample:
    cluster_id: int
    selected: FrameRecord
    cluster_size: int
    candidate_count: int
    avg_hamming: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample keyframes from videos using pHash and agglomerative clustering."
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--frame-dir", type=Path, default=DEFAULT_FRAME_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--frame-manifest",
        type=Path,
        default=None,
        help="Optional existing frame manifest to resample from instead of analyzing videos.",
    )
    return parser.parse_args()


def frame_to_pil_rgb(frame_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def compute_phash_u64(
    frame_bgr: np.ndarray,
    hash_size: int = 8,
    resize_wh: tuple[int, int] = (128, 128),
) -> int:
    """
    Compute pHash (64-bit when hash_size=8) and return as uint64 int.
    """
    w, h = resize_wh
    small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    ph = imagehash.phash(frame_to_pil_rgb(small), hash_size=hash_size)

    bits = ph.hash.flatten().astype(np.uint8)
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def var_laplacian(frame_bgr: np.ndarray) -> float:
    """
    Variance of Laplacian on grayscale. Higher => sharper.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def hamming_u64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def extract_frame_records(
    video_path: Path,
    frame_stride: int = 1,
    hash_size: int = 8,
) -> list[FrameRecord]:
    cap = cv2.VideoCapture(video_path)

    records: list[FrameRecord] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_stride == 0:
            phash = compute_phash_u64(frame, hash_size=hash_size)
            blur = var_laplacian(frame)
            video_idx = int(video_path.stem.partition("_")[0])

            records.append(
                FrameRecord(
                    video_path=video_path,
                    video_idx=video_idx,
                    frame_idx=frame_idx,
                    phash=phash,
                    blur_score=blur,
                    frame_bgr=frame.copy(),
                    height=frame.shape[0],
                    width=frame.shape[1],
                )
            )

        frame_idx += 1

    cap.release()
    return records


def phash_distance_matrix(records: list[FrameRecord]) -> NDArray:
    n = len(records)
    dists = np.zeros((n, n), dtype=float)

    for i in range(n):
        hash_i = records[i].phash
        for j in range(i + 1, n):
            hash_j = records[j].phash
            dist = hamming_u64(hash_i, hash_j)
            dists[i, j] = dist
            dists[j, i] = dist

    return dists


def cluster_phash(
    records: list[FrameRecord],
    n_samples: int | None = None,
    hamming_threshold: int | None = 12,
    linkage: Literal["complete", "average", "single"] = "complete",
) -> list[list[int]]:
    n = len(records)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    dists = phash_distance_matrix(records)

    if n_samples is not None:
        hamming_threshold = None

    model = AgglomerativeClustering(
        n_clusters=n_samples,
        metric="precomputed",
        linkage=linkage,
        distance_threshold=hamming_threshold,
    )

    model.fit(dists)

    labels = model.labels_
    sets: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        sets.setdefault(int(label), []).append(i)

    clusters = list(sets.values())

    # Sort clusters by size
    clusters.sort(
        key=lambda idxs: (-len(idxs), min(records[i].frame_idx for i in idxs))
    )
    return clusters


def filtered_medoid(
    records: list[FrameRecord], cluster_idxs: list[int], blur_filter: float = 0.3
) -> tuple[FrameRecord, int, float]:
    """
    Filter blurry frames and return frame with lowest average Hamming distance
    within a single cluster.

    Args:
        records (List[FrameRecord]): _description_
        cluster_idxs (List[int]): _description_
        blur_filter (float, optional): _description_. Defaults to 0.3.

    Returns:
        Tuple[FrameRecord, int, float]: _description_
    """
    cluster_sorted = sorted(
        cluster_idxs,
        key=lambda i: records[i].blur_score,
        reverse=True,
    )

    candidate_count = max(1, int(round(len(cluster_sorted) * blur_filter)))
    candidate_idxs = cluster_sorted[:candidate_count]

    best_idx: Optional[int] = None
    best_avg = float("inf")

    for candidate_idx in candidate_idxs:
        candidate_hash = records[candidate_idx].phash

        distances = [
            hamming_u64(candidate_hash, records[other_idx].phash)
            for other_idx in cluster_idxs
            if other_idx != candidate_idx
        ]

        avg_dist = float(np.mean(distances)) if distances else 0.0

        if best_idx is None:
            best_idx = candidate_idx
            best_avg = avg_dist
            continue

        current = records[candidate_idx]
        best = records[best_idx]

        is_better = (
            avg_dist < best_avg
            or (avg_dist == best_avg and current.blur_score > best.blur_score)
            or (
                avg_dist == best_avg
                and current.blur_score == best.blur_score
                and current.frame_idx < best.frame_idx
            )
        )

        if is_better:
            best_idx = candidate_idx
            best_avg = avg_dist

    assert best_idx is not None
    return records[best_idx], candidate_count, best_avg


def sample_video(
    video_path: Path,
    hamming_threshold: int = 12,
    blur_filter: float = 0.3,
    frame_stride: int = 1,
    hash_size: int = 8,
    n_samples: int | None = None,
    linkage: Literal["complete", "average", "single"] = "complete",
) -> list[ClusterSample]:
    records = extract_frame_records(
        video_path=video_path,
        frame_stride=frame_stride,
        hash_size=hash_size,
    )

    clusters = cluster_phash(
        records=records,
        n_samples=n_samples,
        hamming_threshold=hamming_threshold,
        linkage=linkage,
    )

    samples: list[ClusterSample] = []

    for cluster_id, cluster_idxs in enumerate(clusters):
        medoid, candidate_count, avg_hamming = filtered_medoid(
            records=records,
            cluster_idxs=cluster_idxs,
            blur_filter=blur_filter,
        )

        samples.append(
            ClusterSample(
                cluster_id=cluster_id,
                selected=medoid,
                cluster_size=len(cluster_idxs),
                candidate_count=candidate_count,
                avg_hamming=avg_hamming,
            )
        )

    # Sort keyframes by order
    samples.sort(key=lambda cs: cs.selected.frame_idx)

    return samples


def build_manifest_id(video_idx: int, cluster_id: int, frame_idx: int) -> int:
    return (video_idx + 1) * int(1e10) + (cluster_id + 1) * int(1e6) + frame_idx


def build_output_path(
    output_dir: Path,
    video_path: Path,
    frame_idx: int,
    cluster_id: int,
    img_ext: str,
) -> Path:
    return (
        output_dir
        / video_path.stem
        / f"frame-{frame_idx:06d}_cluster-{cluster_id:04d}{img_ext}"
    )


def build_manifest_row(
    *,
    video_path: Path,
    height: int,
    width: int,
    video_id: int,
    cluster_id: int,
    frame_idx: int,
    blur_score: float,
    cluster_size: int,
    candidate_count: int,
    avg_hamming: float,
    output_path: Path,
) -> dict[str, str]:
    return {
        "video_path": str(video_path),
        "id": str(build_manifest_id(video_id, cluster_id, frame_idx)),
        "height": str(height),
        "width": str(width),
        "video_id": str(video_id),
        "cluster_id": str(cluster_id),
        "frame_idx": str(frame_idx),
        "blur_score": f"{blur_score:.4f}",
        "cluster_size": str(cluster_size),
        "candidate_count": str(candidate_count),
        "avg_hamming": f"{avg_hamming:.4f}",
        "output_path": str(output_path),
    }


def write_manifest(rows: list[dict[str, str]], manifest_dir: Path) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "frame_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FRAME_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_samples(
    samples: list[ClusterSample],
    output_dir: Path,
    manifest_dir: Path,
    img_ext: str = ".jpg",
    jpeg_quality: int = 95,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []

    for sample in samples:
        record = sample.selected
        output_path = build_output_path(
            output_dir=output_dir,
            video_path=record.video_path,
            frame_idx=record.frame_idx,
            cluster_id=sample.cluster_id,
            img_ext=img_ext,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if img_ext.lower() in (".jpg", ".jpeg"):
            cv2.imwrite(
                str(output_path),
                record.frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
        else:
            cv2.imwrite(str(output_path), record.frame_bgr)

        manifest_rows.append(
            build_manifest_row(
                video_path=record.video_path,
                height=record.height,
                width=record.width,
                video_id=record.video_idx,
                cluster_id=sample.cluster_id,
                frame_idx=record.frame_idx,
                blur_score=record.blur_score,
                cluster_size=sample.cluster_size,
                candidate_count=sample.candidate_count,
                avg_hamming=sample.avg_hamming,
                output_path=output_path,
            )
        )

    write_manifest(manifest_rows, manifest_dir)


def resolve_manifest_video_path(video_path_value: str, video_dir: Path) -> Path | None:
    manifest_video_path = Path(video_path_value).expanduser()
    candidates = [manifest_video_path]

    if manifest_video_path.parent != Path("."):
        candidates.append(video_dir / manifest_video_path.name)
    else:
        candidates.append(video_dir / manifest_video_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def resample_from_manifest(
    frame_manifest: Path,
    video_dir: Path,
    output_dir: Path,
    manifest_dir: Path,
) -> None:
    with frame_manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    rows_by_video: dict[Path, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    missing_videos: set[str] = set()
    for row_idx, row in enumerate(rows):
        resolved_video_path = resolve_manifest_video_path(row["video_path"], video_dir)
        if resolved_video_path is None:
            missing_videos.add(row["video_path"])
            continue

        rows_by_video[resolved_video_path].append((row_idx, row))

    if missing_videos:
        print(
            "sample_keyframes_warning",
            {
                "missing_videos": sorted(missing_videos),
                "num_missing_videos": len(missing_videos),
            },
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_rows: dict[int, dict[str, str]] = {}
    unreadable_frames: list[dict[str, str | int]] = []

    for video_path, video_rows in rows_by_video.items():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(
                "sample_keyframes_warning",
                {
                    "video_path": str(video_path),
                    "warning": "Unable to open video. Skipping.",
                },
            )
            continue

        for row_idx, row in sorted(
            video_rows, key=lambda item: int(item[1]["frame_idx"])
        ):
            frame_idx = int(row["frame_idx"])
            cluster_id = int(row["cluster_id"])
            output_path_value = row.get("output_path", "")
            img_ext = Path(output_path_value).suffix or ".jpg"
            output_path = build_output_path(
                output_dir=output_dir,
                video_path=video_path,
                frame_idx=frame_idx,
                cluster_id=cluster_id,
                img_ext=img_ext,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                unreadable_frames.append(
                    {"video_path": str(video_path), "frame_idx": frame_idx}
                )
                continue

            if img_ext.lower() in (".jpg", ".jpeg"):
                cv2.imwrite(
                    str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                )
            else:
                cv2.imwrite(str(output_path), frame)

            saved_rows[row_idx] = {
                **row,
                "video_path": str(video_path),
                "output_path": str(output_path),
            }

        cap.release()

    if unreadable_frames:
        print(
            "sample_keyframes_warning",
            {
                "unreadable_frames": unreadable_frames[:10],
                "num_unreadable_frames": len(unreadable_frames),
            },
        )

    ordered_rows = [
        saved_rows[row_idx] for row_idx in range(len(rows)) if row_idx in saved_rows
    ]
    write_manifest(ordered_rows, manifest_dir)
    print(
        "sample_keyframes",
        {
            "manifest_path": str(manifest_dir / "frame_manifest.csv"),
            "num_rows": len(ordered_rows),
            "source_manifest": str(frame_manifest),
        },
    )


def main():
    args = parse_args()
    assert not args.video_dir.is_file()
    assert not args.frame_dir.is_file()
    assert not args.manifest_dir.is_file()

    if args.frame_manifest is not None:
        resample_from_manifest(
            frame_manifest=args.frame_manifest,
            video_dir=args.video_dir,
            output_dir=args.frame_dir,
            manifest_dir=args.manifest_dir,
        )
        return

    # This may get pretty large if there's lots of data, maybe periodically flush buffer to file
    all_samples = []
    for video in args.video_dir.rglob("*.mp4"):
        start = time.perf_counter()
        samples = sample_video(video)
        all_samples.extend(samples)
        end = time.perf_counter()
        print(
            f"Sampled {len(samples)} frames from {video.name} in {end - start:.1f} seconds"
        )

    save_samples(all_samples, args.frame_dir, args.manifest_dir)


if __name__ == "__main__":
    main()
