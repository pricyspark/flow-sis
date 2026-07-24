from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def resolve_video_source(source: str) -> int | str:
    if source == "live":
        return 0
    if source.isdigit():
        return int(source)
    return source


def center_square(frame: NDArray) -> NDArray:
    height, width = frame.shape[:2]
    size = min(height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return np.ascontiguousarray(frame[top : top + size, left : left + size])
