import argparse
import csv
import json
import os
from pathlib import Path
import csv
from typing import Dict, Tuple
from numpy.typing import NDArray
from datasets import Dataset, DatasetDict, Features, Sequence, Value, Image

import numpy as np

def load_binary(path: Path) -> np.ndarray:
    data = np.load(path)
    packed = data["packed"]
    shape = tuple(data["shape"])
    n_bits = np.prod(shape)
    flat = np.unpackbits(packed)[:n_bits]
    arr = flat.reshape(shape).astype(bool)
    return arr

def build_dataset(path: Path, classes: Tuple[Dict[str, str], Dict[str, int]]):
    rows = []
    bboxes: Dict[str, NDArray] = {}
    for file in Path("data/boxes").iterdir():
        if file.stem == ".gitkeep":
            continue
        bboxes[file.stem] = np.load(file)
        
    last_image = ""
    bbox_id = 0
    with open("data/frames/frame_manifest.csv", newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            image, image_id, height, width, video_id = row[:5]
            
            if image != last_image:
                last_image = image
                counter = 0
            
            if video_id not in bboxes:
                print(f"{video_id} does not a generated bounding box")
                continue
            
            bbox = bboxes[video_id][counter]
            class_str = classes[0][video_id]
            class_int = classes[1][class_str]
            
            entry = {
                "image": image,
                "image_id": image_id,
                "height": height,
                "width": width,
                "objects": {
                    "id": [bbox_id],
                    "area": [bbox[2] * bbox[3]],
                    "bbox": [list(bbox)],
                    "category": [class_int],
                },
            }
            
            counter += 1
            bbox_id += 1
            rows.append(entry)
            
    features = Features({
        "image": Image(),
        "image_id": Value("int64"),
        "height": Value("int64"),
        "width": Value("int64"),
        "objects": {
            "id": Sequence(Value("int64")),
            "area": Sequence(Value("float32")),
            "bbox": Sequence(Sequence(Value("float32"), length=4)),
            "category": Sequence(Value("int64")),
        },
    })
    
    dataset = Dataset.from_list(rows, features=features)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    test_val = split["test"].train_test_split(test_size=0.5, seed=42)
    dataset_splits = DatasetDict({
        "train": split["train"],
        "validation": test_val["train"],
        "test": test_val["test"]
    })
    dataset_splits.save_to_disk(path)
            
def main():
    with open("data/raw/classes.json") as file:
        classes = json.load(file)
    build_dataset(Path("data/dataset"), classes)
            
if __name__ == "__main__":
    main()
