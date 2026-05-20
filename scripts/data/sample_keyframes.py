import cv2
import imagehash
import numpy as np
from PIL import Image
from pathlib import Path
from numpy.typing import NDArray
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering
from typing import Optional, Literal
import time


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
            video_idx = int(video_path.stem.partition('_')[0])
            
            records.append(
                FrameRecord(
                    video_path = video_path,
                    video_idx = video_idx,
                    frame_idx = frame_idx,
                    phash = phash,
                    blur_score = blur,
                    frame_bgr = frame.copy(),
                    height = frame.shape[0],
                    width = frame.shape[1],
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
        n_clusters = n_samples,
        metric = "precomputed",
        linkage = linkage,
        distance_threshold = hamming_threshold,
    )
        
    model.fit(dists)
    
    labels = model.labels_
    sets: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        sets.setdefault(int(label), []).append(i)
            
    clusters = list(sets.values())
    
    # Sort clusters by size
    clusters.sort(
        key = lambda idxs: (
            -len(idxs),
            min(records[i].frame_idx for i in idxs)
        )
    )
    return clusters


def filtered_medoid(
    records: list[FrameRecord],
    cluster_idxs: list[int],
    blur_filter: float = 0.3
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
        key = lambda i : records[i].blur_score,
        reverse = True,
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
            or (
                avg_dist == best_avg
                and current.blur_score > best.blur_score
            )
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
        video_path = video_path,
        frame_stride = frame_stride,
        hash_size = hash_size,
    )
    
    clusters = cluster_phash(
        records = records,
        n_samples = n_samples,
        hamming_threshold = hamming_threshold,
        linkage = linkage,
    )
    
    samples: list[ClusterSample] = []
    
    for cluster_id, cluster_idxs in enumerate(clusters):
        medoid, candidate_count, avg_hamming = filtered_medoid(
            records = records,
            cluster_idxs = cluster_idxs,
            blur_filter = blur_filter,
        )
        
        samples.append(
            ClusterSample(
                cluster_id = cluster_id,
                selected = medoid,
                cluster_size = len(cluster_idxs),
                candidate_count = candidate_count,
                avg_hamming = avg_hamming,
            )
        )
    
    # Sort keyframes by order
    samples.sort(key=lambda cs: cs.selected.frame_idx)
    
    return samples


def save_samples(
    samples: list[ClusterSample],
    output_dir: Path,
    img_ext: str = ".jpg",
    jpeg_quality: int = 95,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_rows = [
        "video_path,id,height,width,video_id,cluster_id,frame_idx,blur_score,cluster_size,candidate_count,avg_hamming,output_path"
    ]
    
    for sample in samples:
        record = sample.selected
        
        filename = (
            f"{record.video_path.stem}"
            f"_frame-{record.frame_idx:06d}"
            f"_cluster-{sample.cluster_id:04d}"
            f"{img_ext}"
        )
        
        output_path = output_dir / filename
        
        if img_ext.lower() in (".jpg", ".jpeg"):
            cv2.imwrite(
                str(output_path),
                record.frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
        else:
            cv2.imwrite(str(output_path), record.frame_bgr)
            
        manifest_id = (
            (record.video_idx + 1) * int(1e10)
            + (sample.cluster_id + 1) * int(1e6)
            + record.frame_idx
        )
            
        manifest_rows.append(
            ",".join(
                [
                    str(record.video_path),
                    str(manifest_id),
                    str(record.height),
                    str(record.width),
                    str(record.video_idx),
                    str(sample.cluster_id),
                    str(record.frame_idx),
                    f"{record.blur_score:.4f}",
                    str(sample.cluster_size),
                    str(sample.candidate_count),
                    f"{sample.avg_hamming:.4f}",
                    str(output_path),
                ]
            )
        )
        
    manifest_path = output_dir / "frame_manifest.csv"
    manifest_path.write_text("\n".join(manifest_rows), encoding="utf-8")
    
    
def main():
    video_dir = Path("data/raw")
    frame_dir = Path("data/frames")
    assert not video_dir.is_file()
    assert not frame_dir.is_file()
    
    # This may get pretty large if there's lots of data, maybe periodically flush buffer to file
    all_samples = []
    for video in video_dir.rglob("*.mp4"):
        start = time.perf_counter()
        samples = sample_video(video)
        all_samples.extend(samples)
        end = time.perf_counter()
        print(f"Sampled {len(samples)} frames from {video.name} in {end - start:.1f} seconds")
    
    save_samples(all_samples, frame_dir)
    
if __name__ == "__main__":
    main()