import copy
import numpy as np
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol
from torch.utils.data import Dataset

from flowsis.utils.common import init_rng


class RuntimeCallable(Protocol):
    def __call__(self, x: Any, /, **kwargs: Any) -> Any: ...


class CallablePipeline:
    def __init__(
        self,
        callables: Iterable[RuntimeCallable],
        callable_kwargs: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        self.callables = list(callables)
        self.callable_kwargs = (
            []
            if callable_kwargs is None
            else [dict(kwargs) for kwargs in callable_kwargs]
        )
        self.callable_kwargs += [
            {} for _ in range(len(self.callables) - len(self.callable_kwargs))
        ]

    def __call__(self, x: Any, **runtime_kwargs: Any) -> Any:
        for callable_, callable_kwarg in zip(self.callables, self.callable_kwargs):
            kwargs = {**runtime_kwargs, **callable_kwarg}
            result = callable_(x, **kwargs)
            if result is not None:
                x = result
        return x

    def __len__(self) -> int:
        return len(self.callables)

    def append(
        self, callable_: RuntimeCallable, kwargs: dict[str, Any] | None = None
    ) -> None:
        self.callables.append(callable_)
        self.callable_kwargs.append({} if kwargs is None else dict(kwargs))


@dataclass(frozen=True)
class SampleContext:
    dataset: Any
    index: int
    loader: Callable[[dict[str, Any]], Any] | None = None
    rng: np.random.Generator | None = None
    seed: int | None = None
    copy_examples: bool = True

    def __len__(self) -> int:
        return len(self.dataset)

    def _prepare_example(self, example: dict[str, Any]) -> dict[str, Any]:
        prepared = copy.deepcopy(example) if self.copy_examples else example
        if self.loader is not None:
            loaded = self.loader(prepared)
            if loaded is not None:
                prepared = loaded
        return prepared

    def get_example(self, index: int) -> dict[str, Any]:
        return self._prepare_example(self.dataset[index])

    def get_relative_example(
        self,
        offset: int,
        *,
        wrap: bool = True,
    ) -> dict[str, Any]:
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
        rng: np.random.Generator | None = None,
    ) -> list[dict[str, Any]]:
        if count <= 0:
            return []

        if rng is None:
            rng = init_rng(self.rng, self.seed)
        candidate_indices = list(range(len(self)))
        if exclude_current:
            candidate_indices = [idx for idx in candidate_indices if idx != self.index]
        if not candidate_indices:
            return []

        if replace:
            selected_indices = rng.choice(candidate_indices, size=count, replace=True)
        else:
            selected_count = min(count, len(candidate_indices))
            selected_indices = rng.choice(
                candidate_indices,
                size=selected_count,
                replace=False,
            )

        return [self.get_example(idx) for idx in selected_indices]


class PreparedDataset(Dataset):
    def __init__(
        self,
        base_dataset: Any,
        loader: RuntimeCallable | None = None,
        augment: RuntimeCallable | None = None,
        *,
        copy_examples: bool = True,
        context_factory: type[SampleContext] = SampleContext,
        seed: int | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.loader = loader
        self.augment = augment
        self.copy_examples = copy_examples
        self.context_factory = context_factory
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base_dataset)

    def get_raw_example(self, idx: int) -> dict[str, Any]:
        example = self.base_dataset[idx]
        return copy.deepcopy(example) if self.copy_examples else example

    def load_example(self, example: dict[str, Any]) -> dict[str, Any]:
        if self.loader is None:
            return example

        loaded = self.loader(example)
        return example if loaded is None else loaded

    def build_context(self, idx: int) -> SampleContext:
        rng = None if self.seed is None else np.random.default_rng(self.seed + idx)
        return self.context_factory(
            dataset=self.base_dataset,
            index=idx,
            loader=self.loader,
            rng=rng,
            copy_examples=self.copy_examples,
        )

    def __getitem__(self, idx: int) -> Any:
        example = self.get_raw_example(idx)
        example = self.load_example(example)

        if self.augment is None:
            return example

        context = self.build_context(idx)
        augmented = self.augment(example, context=context)
        return example if augmented is None else augmented
