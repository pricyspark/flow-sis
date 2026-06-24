from pathlib import Path
from typing import Any

from PIL import Image


def _open_image(path: str | Path) -> Image.Image:
    resolved_path = Path(path).expanduser()
    with Image.open(resolved_path) as img:
        return img.copy()


def get_example_image_source(example: dict[str, Any]) -> str | None:
    image_path = example.get("image_path")
    if image_path is not None:
        return str(image_path)

    image_value = example.get("image")
    if isinstance(image_value, Image.Image):
        return "<loaded-image>"
    return None


def _load_example_image(
    example: dict[str, Any],
    *,
    convert_mode: str | None = None,
) -> Image.Image:
    image_value = example.get("image")

    if isinstance(image_value, Image.Image):
        image = image_value.copy()
    elif "image_path" in example:
        image = _open_image(example["image_path"])
    else:
        raise KeyError("Example does not contain an image or image_path field.")

    if convert_mode is not None and image.mode != convert_mode:
        image = image.convert(convert_mode)
    return image


def get_image(
    example: dict[str, Any],
    *,
    convert_mode: str | None = None,
) -> Image.Image:
    image_value = example.get("image")
    if isinstance(image_value, Image.Image):
        if convert_mode is not None and image_value.mode != convert_mode:
            image_value = image_value.convert(convert_mode)
            example["image"] = image_value
        return image_value

    image = _load_example_image(example, convert_mode=convert_mode)
    example["image"] = image
    return image
