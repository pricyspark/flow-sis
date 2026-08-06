import numpy as np
from numpy.typing import NDArray
from pathlib import Path

DEFAULT_MASKS_DIR = Path("data/masks")


def load_binary(
    path,
) -> NDArray[np.bool_]:  # TODO: this should go in a general utils file
    data = np.load(path)
    packed = data["packed"]
    shape = tuple(data["shape"])
    n_bits = np.prod(shape)
    flat = np.unpackbits(packed)[:n_bits]
    arr = flat.reshape(shape).astype(bool)
    return arr


def load_mask(video_id: int, frame_idx: int, path=None) -> NDArray[np.bool_]:
    if path is None:
        path = DEFAULT_MASKS_DIR
    mask_path = path / f"{video_id}" / f"{frame_idx}.npz"
    mask = load_binary(mask_path)
    return mask


def mask2xywh(mask: NDArray) -> list[int] | None:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return None

    x_min = cols.min()
    x_max = cols.max()
    y_min = rows.min()
    y_max = rows.max()

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    return [
        int(x_min),
        int(y_min),
        int(width),
        int(height),
    ]
