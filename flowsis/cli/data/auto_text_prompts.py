import os
import argparse
from openai import OpenAI
from pydantic import BaseModel, Field
from pathlib import Path

from flowsis.utils import load_classes
from flowsis.data import LabelPrompts

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_CLASSES_PATH = Path("data/manifests/classes.json")
DEFAULT_PROMPTS_PATH = Path("data/manifests/auto_text_prompts.json")
DEFAULT_EMBEDDINGS_PATH = Path("data/manifests/text-embeddings")
with open("auto_prompt.txt", "r") as f:
    INSTRUCTIONS = f.read().strip()


openai_key = os.environ.get("OPENAI_API_KEY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use the OpenAI API to remotely generate text prompts for each label class."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--classes-path", type=Path, default=DEFAULT_CLASSES_PATH)
    parser.add_argument("--prompts-path", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--embeddings-path", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    return parser.parse_args()


class InstrumentPrompt(BaseModel):
    enriched_phrase: str = Field(
        description="lowercase visual phrase under 12 words, starting with the instrument name when possible"
    )
    visual_discriminators: list[str] = Field(
        description="2 to 5 short visible features, each 1 to 4 words"
    )


def main():
    if openai_key is None:
        print(
            "Environment variable OPENAI_API_KEY is not set.\n"
            "OPENAI_API_KEY is required to make requests for text prompt autogeneration.\n"
            "Visit https://platform.openai.com/api-keys to generate a key if you don't already one."
        )
        return

    args = parse_args()
    client = OpenAI(api_key=openai_key)
    vid2label, label2id, id2label = load_classes(args.classes_path)
    label_prompts = LabelPrompts()

    for label in label2id.keys():
        response = client.responses.parse(
            model=args.model,
            instructions=INSTRUCTIONS,
            input=f"Instrument name: {label}",
            text_format=InstrumentPrompt,
            temperature=0,
        )
        assert (prompt := response.output_parsed)

        label_prompts.add(label, prompt.enriched_phrase)
        print(f"Prompt generated for: {label}")

    label_prompts.dump(args.prompts_path)
    label_prompts.embed_all()
    label_prompts.save_embeddings(args.embeddings_path)


if __name__ == "__main__":
    main()


"""
load_boxes
build_dataset
create_json

"""
