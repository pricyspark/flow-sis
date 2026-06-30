from pathlib import Path
from PIL import Image
from typing import Any

from ..data.masks import load_mask


def load_object_image(example: dict[str, Any], **kwargs) -> dict:
    if "image" in example:
        return example
    
    if example["modified"]:
        raise ValueError("Cannot load an image for a previously modified sample.")
    
    image_path = example["image_path"]
    example["image"] = Image.open(image_path)
    
    return example


def load_object_masks(example: dict[str, Any], **kwargs) -> dict:
    if "mask_dir" in kwargs:
        mask_dir = Path(kwargs["mask_dir"])
    else:
        mask_dir = None
    
    for obj in example["objects"]:
        if "mask" in obj:
            continue
        
        if obj["modified"]:
            raise ValueError("Cannot load a mask for a previously modified object.")
        
        video_id = obj["video_id"]
        frame_idx = obj["frame_idx"]
        
        obj["mask"] = load_mask(video_id, frame_idx, mask_dir)
    
    return example
