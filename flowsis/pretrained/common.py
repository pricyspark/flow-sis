from pathlib import Path


def resolve_pretrained_source(model_name_or_path: str) -> tuple[str, bool]:
    """Return a local source unchanged and let Hugging Face resolve remote caches."""
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        return str(path.resolve()), True
    return model_name_or_path, False
