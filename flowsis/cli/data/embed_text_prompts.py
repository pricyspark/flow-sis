import argparse
import json
from pathlib import Path

import torch

from flowsis.pretrained.siglip2 import SigLIP2
from flowsis.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache one pooled SigLIP2 embedding for each stored text prompt."
    )
    parser.add_argument(
        "--prompts_path",
        type=Path,
        default=Path("data/manifests/manual_text_prompts.json"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/manifests/text-embeddings"),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="google/siglip2-base-patch16-224",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.prompts_path.open() as file:
        prompts = json.load(file)
    if not isinstance(prompts, dict):
        raise TypeError("The prompt manifest must be a label-to-prompt-list mapping.")

    device = torch.device(args.device) if args.device is not None else get_device()
    encoder = SigLIP2.from_pretrained(
        args.model_name_or_path,
        return_mode="pooled",
        normalize=True,
        device=device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for label, label_prompts in prompts.items():
        if not isinstance(label, str) or not isinstance(label_prompts, list) or not label_prompts:
            raise ValueError(f"Invalid prompt entry for {label!r}.")
        embeddings = encoder([str(prompt) for prompt in label_prompts]).cpu()
        torch.save(embeddings, args.output_dir / f"{label}.pt")
        print(label, tuple(embeddings.shape))


if __name__ == "__main__":
    main()
