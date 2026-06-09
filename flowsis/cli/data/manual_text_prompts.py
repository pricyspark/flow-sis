import argparse
from pathlib import Path

from flowsis.utils import load_classes
from flowsis import LabelPrompts


DEFAULT_CLASSES_PATH = Path("data/manifests/classes.json")
DEFAULT_PROMPTS_PATH = Path("data/manifests/manual_text_prompts.json")
DEFAULT_EMBEDDINGS_PATH = Path("data/manifests/text-embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually enter text prompts for each label class.")
    parser.add_argument("--classes_path", type=Path, default=DEFAULT_CLASSES_PATH)
    parser.add_argument("--prompts_path", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--embeddings_path", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    return parser.parse_args()


def main():    
    args = parse_args()
    vid2label, label2id, id2label = load_classes(args.classes_path)
    label_prompts = LabelPrompts()
    
    for label in label2id.keys():
        prompt = input(f"Please enter the prompt for the label class {label}:\n").strip()
        label_prompts.add(label, prompt)
        print(f"Prompt generated for: {label}")
        
    label_prompts.dump(args.prompts_path)
    label_prompts.save_embeddings(args.embeddings_path)
        
if __name__ == "__main__":
    main()
