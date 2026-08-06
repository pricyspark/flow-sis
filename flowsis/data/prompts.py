from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import torch

if TYPE_CHECKING:
    from flowsis.pretrained.siglip2 import SigLIP2


def apply_template(label: str) -> str:
    return f"This is a photo of {label}."


class LabelPrompts:
    def __init__(
        self,
        *,
        model_name_or_path: str = "google/siglip2-base-patch16-224",
        device: str | torch.device | None = None,
    ) -> None:
        self.prompts: dict[str, list[str]] = {}
        self.embeddings: dict[str, torch.Tensor] = {}
        self.model_name_or_path = model_name_or_path
        self.device = device
        self._siglip2: SigLIP2 | None = None

    @property
    def siglip2(self) -> SigLIP2:
        """Load the text encoder only when embeddings are requested."""
        if self._siglip2 is None:
            from flowsis.pretrained.siglip2 import SigLIP2

            self._siglip2 = SigLIP2.from_pretrained(
                self.model_name_or_path,
                return_mode="pooled",
                normalize=True,
                device=self.device,
            )
        return self._siglip2

    def add(self, label: str, prompt: str) -> None:
        if label in self.prompts:
            warnings.warn(f"Label {label} already has prompts. Overwriting.")

        self.prompts[label] = [
            label,
            apply_template(label),
            prompt,
        ]
        self.embeddings.pop(label, None)

    @classmethod
    def load(
        cls,
        input_path: PathLike[str],
        *,
        model_name_or_path: str = "google/siglip2-base-patch16-224",
        device: str | torch.device | None = None,
    ) -> "LabelPrompts":
        input_path = Path(input_path)
        with input_path.open(encoding="utf-8") as file:
            prompts = json.load(file)

        if not isinstance(prompts, dict):
            raise ValueError(f"Expected a JSON object in {input_path}")

        label_prompts = cls(model_name_or_path=model_name_or_path, device=device)
        for label, prompt_list in prompts.items():
            if not isinstance(label, str) or not isinstance(prompt_list, list):
                raise ValueError(
                    f"Expected each entry in {input_path} to map a label to a list of prompts"
                )
            if not prompt_list or not all(
                isinstance(prompt, str) for prompt in prompt_list
            ):
                raise ValueError(
                    f"Prompts for {label!r} must be a non-empty list of strings"
                )
            label_prompts.prompts[label] = prompt_list

        return label_prompts

    @staticmethod
    def load_embeddings(
        label: str,
        directory: Path,
        cache: dict[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if label not in cache:
            path = directory / f"{label}.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing prompt embeddings for detector label {label!r}: {path}"
                )
            embeddings = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
                shape = getattr(embeddings, "shape", None)
                raise ValueError(
                    f"Expected prompt embeddings shaped [P,D] at {path}, got {shape}."
                )
            cache[label] = embeddings.float()
        # Tensor.to is a no-op when the cached tensor is already on the target
        # device, and migrates it if the caller moved the model since caching.
        cache[label] = cache[label].to(device)
        return cache[label]

    def embed_all(self) -> dict[str, torch.Tensor]:
        """Create one normalized, pooled semantic vector per prompt."""
        self.embeddings = {
            label: self.siglip2(texts=prompts)
            for label, prompts in self.prompts.items()
        }
        return self.embeddings

    def dump(
        self,
        output_path: PathLike[str],
        *,
        indent: int | str | None = None,
        separators: tuple[str, str] | None = None,
        sort_keys: bool = False,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                self.prompts,
                file,
                indent=indent,
                separators=separators,
                sort_keys=sort_keys,
            )

    def dumps(
        self,
        *,
        indent: int | str | None = None,
        separators: tuple[str, str] | None = None,
        sort_keys: bool = False,
    ) -> str:
        return json.dumps(
            self.prompts,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    def save_embeddings(self, embedding_dir: PathLike[str]) -> None:
        embedding_dir = Path(embedding_dir)
        embedding_dir.mkdir(parents=True, exist_ok=True)
        for label, embedding in self.embeddings.items():
            output_path = embedding_dir / f"{label}.pt"
            if output_path.parent != embedding_dir:
                raise ValueError(
                    f"Label {label!r} cannot be used as an embedding filename"
                )
            torch.save(embedding.cpu(), output_path)
