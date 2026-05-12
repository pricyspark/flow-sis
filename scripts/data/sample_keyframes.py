import os
import cv2
import random
import shutil
import argparse
import imagehash
import numpy as np
from PIL import Image
from pathlib import Path
from numpy.typing import NDArray
from dataclasses import dataclass, field
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, List, Optional, Set, Tuple, Literal

@dataclass
class FrameRecord:
    video_path: Path
    frame_idx: int
    phash: int
    blur_score: float
    frame_bgr: NDArray
    
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
    resize_wh: Tuple[int, int] = (128, 128),
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
) -> List[FrameRecord]:
    cap = cv2.VideoCapture(video_path)
    
    records: List[FrameRecord] = []
    
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        if frame_idx % frame_stride == 0:
            phash = compute_phash_u64(frame, hash_size=hash_size)
            blur = var_laplacian(frame)
            
            records.append(FrameRecord(
                video_path = video_path,
                frame_idx = frame_idx,
                phash = phash,
                blur_score = blur,
                frame_bgr = frame.copy()
            ))
            
        frame_idx += 1
    
    cap.release()
    return records

def phash_distance_matrix(records: List[FrameRecord]) -> NDArray:
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
    records: List[FrameRecord],
    n_samples: int | None = None,
    hamming_threshold: int | None = 12,
    linkage: Literal["complete", "average", "single"] = "complete",
) -> List[List[int]]:
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
    sets: Dict[int, List[int]] = {}
    for i, label in enumerate(labels):
        sets.setdefault(int(label), []).append(i)
            
    clusters = list(sets.values())
    clusters.sort(key=lambda idxs: (-len(idxs), min(records[i].frame_idx for i in idxs)))
    return clusters

def filtered_medoid(
    records: List[FrameRecord],
    cluster_idxs: List[int],
    blur_filter: float = 0.3
) -> Tuple[FrameRecord, int, float]:
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
) -> List[ClusterSample]:
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
    
    samples: List[ClusterSample] = []
    
    for cluster_id, cluster_idxs in enumerate(clusters):
        medoid, candidate_count, avg_hamming = filtered_medoid(
            records = records,
            cluster_idxs = cluster_idxs,
            blur_filter = blur_filter,
        )
        
        samples.append(ClusterSample(
            cluster_id = cluster_id,
            selected = medoid,
            cluster_size = len(cluster_idxs),
            candidate_count = candidate_count,
            avg_hamming = avg_hamming,
        ))
        
    return samples

def save_samples(
    samples: List[ClusterSample],
    output_dir: Path,
    jpeg_quality: int = 95,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_rows = [
        "video_path,cluster_id,frame_idx,blur_score,cluster_size,candidate_count,avg_hamming,output_path"
    ]
    
    for sample in samples:
        record = sample.selected
        
        
    
def main():
    print(len(sample_video(Path('data/raw/2.mp4'))))

if __name__ == "__main__":
    main()