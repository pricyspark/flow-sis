import argparse
from pathlib import Path

import torch

from flowsis.data import LabelPrompts
from flowsis.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache one pooled SigLIP2 embedding for each stored text prompt."
    )
    parser.add_argument(
        "--prompts-path",
        type=Path,
        default=Path("data/manifests/manual_text_prompts.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests/text-embeddings"),
    )
    parser.add_argument(
        "--model",
        dest="model_source",
        default="google/siglip2-base-patch16-224",
        metavar="MODEL",
        help="Hugging Face Hub model ID or local model path.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else get_device()
    label_prompts = LabelPrompts.load(
        args.prompts_path,
        model_name_or_path=args.model_source,
        device=device,
    )
    embeddings = label_prompts.embed_all()
    label_prompts.save_embeddings(args.output_dir)
    for label, embedding in embeddings.items():
        print(label, tuple(embedding.shape))


if __name__ == "__main__":
    main()
