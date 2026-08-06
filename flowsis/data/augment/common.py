import numpy as np
from numpy.typing import NDArray


def bounding_box_union(bboxes: NDArray) -> tuple[float, float, float, float]:
    x1 = np.min(bboxes[:, 0])
    y1 = np.min(bboxes[:, 1])

    x2 = np.max(bboxes[:, 0] + bboxes[:, 2])
    y2 = np.max(bboxes[:, 1] + bboxes[:, 3])

    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def mask_union(masks: NDArray[np.bool_]) -> NDArray[np.bool_]:
    if masks.ndim != 3:
        raise ValueError("Masks must to passed as a 3D NumPy array.")

    return np.any(masks, axis=0)
