import csv
import json
import argparse
import warnings
from pathlib import Path
from collections.abc import Iterable


DEFAULT_CSV_PATH = Path("data/manifests/video_manifest.csv")
DEFAULT_JSON_PATH = Path("data/manifests/classes.json")
DEFAULT_COLS = ["ToolStyle", "Class"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read video manifest to create class mapping.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument(
        "--columns",
        nargs="+",
        default=DEFAULT_COLS,
        help="Manifest columns to combine into each class label.",
    )
    return parser.parse_args()


def create_json(
    csv_path: Path, 
    json_path: Path, 
    class_cols: Iterable[int | str]
) -> None:
    vid2label: dict[str, str] = {}
    label2id: dict[str, int] = {}
    id2label: dict[int, str] = {}
    output = (vid2label, label2id, id2label)
    with open(csv_path, newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            components = [
                value
                for col in class_cols
                if (value := row[col] if isinstance(col, int) else row[header.index(col)])
            ]
            
            # TODO: cast to tuple for easy comparison, raise warning or error if different rows map to same string
            
            if not components:
                warnings.warn(f"Video {row[0]} has empty cells. Skipping.")
                continue
            
            c = " ".join(components)
            vid2label[row[0]] = c
            if c not in label2id:
                label2id[c] = len(label2id)
                id2label[len(id2label)] = c
            
    with open(json_path, 'w') as file:
        json.dump(output, file, indent=2)
        
        
def main():
    args = parse_args()
    create_json(
        args.csv_path,
        args.json_path,
        args.columns,
    )
    
    
if __name__ == "__main__":
    main()
