import copy
import random
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Iterable

from torch.utils.data import Dataset

from ..images import get_image


DEFAULT_MASKS_DIR = Path("data/masks")


@dataclass(frozen=True)
class AugmentationContext:
    dataset: Any
    index: int

    def __len__(self) -> int:
        return len(self.dataset)

    def get_example(self, index: int) -> dict[str, Any]:
        example = copy.deepcopy(self.dataset[index])
        get_image(example)
        return example

    def get_relative_example(self, offset: int, *, wrap: bool = True) -> dict[str, Any]:
        if len(self) == 0:
            raise IndexError("Cannot fetch a relative example from an empty dataset.")
        target_index = self.index + offset
        if wrap:
            target_index %= len(self)
        elif target_index < 0 or target_index >= len(self):
            raise IndexError(
                f"Relative offset {offset} from index {self.index} is out of bounds for length {len(self)}."
            )
        return self.get_example(target_index)

    def sample_examples(
        self,
        count: int,
        *,
        exclude_current: bool = True,
        replace: bool = False,
        rng: random.Random | None = None,
    ) -> list[dict[str, Any]]:
        if count <= 0:
            return []

        generator = rng if rng is not None else random
        candidate_indices = list(range(len(self)))
        if exclude_current:
            candidate_indices = [idx for idx in candidate_indices if idx != self.index]
        if not candidate_indices:
            return []

        if replace:
            selected_indices = [generator.choice(candidate_indices) for _ in range(count)]
        else:
            selected_count = min(count, len(candidate_indices))
            selected_indices = generator.sample(candidate_indices, k=selected_count)

        return [self.get_example(idx) for idx in selected_indices]


class TransformDataset(Dataset):
    def __init__(self, base_dataset, transform: Callable):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        example = copy.deepcopy(self.base_dataset[idx])
        get_image(example)
        context = AugmentationContext(self.base_dataset, idx)
        return self.transform(example, augmentation_context=context)


class AugmentationPipeline:
    def __init__(
        self,
        augments: Iterable[Callable],
        augment_kwargs: Iterable[Any],
    ):
        self.augments = list(augments)
        self.augment_kwargs = [dict(kwargs) for kwargs in augment_kwargs]

    def __call__(
        self,
        example: dict,
        augmentation_context: AugmentationContext | None = None,
    ):
        for augment, augment_kwargs in zip(self.augments, self.augment_kwargs):
            current_kwargs = dict(augment_kwargs)
            if augmentation_context is not None:
                current_kwargs.setdefault("augmentation_context", augmentation_context)
            example = augment(example, **current_kwargs)
        return example

    def __len__(self) -> int:
        return len(self.augments)

    def append(self, augment: Callable, kwargs: dict[str, Any] | None = None) -> None:
        self.augments.append(augment)
        self.augment_kwargs.append({} if kwargs is None else dict(kwargs))
