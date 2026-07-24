import argparse
from pathlib import Path

import torch

from flowsis.data.prompts import LabelPrompts
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
    device = torch.device(args.device) if args.device is not None else get_device()
    label_prompts = LabelPrompts.load(
        args.prompts_path,
        model_name_or_path=args.model_name_or_path,
        device=device,
    )
    embeddings = label_prompts.embed_all()
    label_prompts.save_embeddings(args.output_dir)
    for label, embedding in embeddings.items():
        print(label, tuple(embedding.shape))


if __name__ == "__main__":
    main()
