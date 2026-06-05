import json
import torch
import random
import numpy as np
from pathlib import Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_pretrained_source(model_name_or_path: str, cache_dir: str) -> tuple[str, bool]:
    """Resolve whether pretrained source is local or must be installed.

    Args:
        model_name_or_path (str): _description_
        cache_dir (str): _description_

    Returns:
        tuple[str, bool]: _description_
    """
    path = Path(model_name_or_path)
    if path.exists():
        return str(path), True

    repo_dir = Path(cache_dir) / f"models--{model_name_or_path.replace('/', '--')}"
    if not repo_dir.exists():
        return model_name_or_path, False

    refs_dir = repo_dir / "refs"
    if refs_dir.exists():
        for ref_file in refs_dir.iterdir():
            commit = ref_file.read_text().strip()
            snapshot_dir = repo_dir / "snapshots" / commit
            if snapshot_dir.exists():
                return str(snapshot_dir), True

    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshot for snapshot in snapshots_dir.iterdir() if snapshot.is_dir())
        if snapshots:
            return str(snapshots[-1]), True

    return model_name_or_path, False

def load_classes(
    classes_path: Path,
) -> tuple[
    dict[str, str],
    dict[str, int],
    dict[int, str],
]:
    with classes_path.open() as file:
        classes = json.load(file)

    vid2label, label2id, raw_id2label = classes
    id2label = {int(key): value for key, value in raw_id2label.items()}
    return vid2label, label2id, id2label
