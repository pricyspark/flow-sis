import json
import torch
import warnings
from os import PathLike
from pathlib import Path

from .pretrained.siglip2 import SigLIP2 # TODO: refactor file structure and go back to absolute imports
from .utils.common import get_device # TODO: refactor file structure and go back to absolute imports


def apply_template(label: str) -> str:
    return f"This is a photo of {label}."


class LabelPrompts:
    def __init__(self) -> None:
        self.prompts: dict[str, list[str]] = {}
        self.embeddings: dict[str, torch.Tensor] = {}
        self.siglip2 = SigLIP2(device=get_device())

    def add(self, label: str, prompt: str) -> None:
        if label in self.prompts:
            warnings.warn(f"Label {label} already has prompts. Overwriting.")
            
        self.prompts[label] = [
            label,
            apply_template(label),
            prompt,
        ]
        self.embeddings[label] = self.siglip2(texts=self.prompts[label])
        
    def dump(self, output_path: PathLike, *, indent=None, separators=None, sort_keys=False) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w") as fp:
            json.dump(
                self.prompts,
                fp,
                indent=indent,
                separators=separators,
                sort_keys=sort_keys,
            )
    
    def dumps(self, *, indent=None, separators=None, sort_keys=False) -> str:
        return json.dumps(
            self.prompts, 
            indent=indent, 
            separators=separators, 
            sort_keys=sort_keys
        ) 

    def save_embeddings(self, embedding_dir: Path) -> None:
        embedding_dir.mkdir(parents=True, exist_ok=True)
        for label, embedding in self.embeddings.items():
            torch.save(embedding, embedding_dir / f"{label}.pt")
