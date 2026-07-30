from pathlib import Path

import torch

from flowsis.data.prompts import LabelPrompts


def test_load_embeddings_moves_cached_tensor_to_requested_device(
    tmp_path: Path,
) -> None:
    torch.save(torch.ones(2, 3), tmp_path / "object.pt")
    cache: dict[str, torch.Tensor] = {}

    cpu_embeddings = LabelPrompts.load_embeddings(
        "object",
        tmp_path,
        cache,
        device=torch.device("cpu"),
    )
    meta_embeddings = LabelPrompts.load_embeddings(
        "object",
        tmp_path,
        cache,
        device=torch.device("meta"),
    )

    assert cpu_embeddings.device.type == "cpu"
    assert meta_embeddings.device.type == "meta"
    assert cache["object"] is meta_embeddings
