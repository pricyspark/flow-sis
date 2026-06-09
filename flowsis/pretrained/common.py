from pathlib import Path


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
