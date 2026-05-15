import argparse
import csv
import json
import os
from pathlib import Path
import csv
from typing import Dict
from numpy.typing import NDArray
from datasets import Dataset, Features, Sequence, Value, Image

import numpy as np


def load_binary(path: Path) -> np.ndarray:
    data = np.load(path)
    packed = data["packed"]
    shape = tuple(data["shape"])
    n_bits = np.prod(shape)
    flat = np.unpackbits(packed)[:n_bits]
    arr = flat.reshape(shape).astype(bool)
    return arr

def build_dataset(path):
    rows = []
    bboxes: Dict[str, NDArray] = {}
    for file in Path("data/boxes").iterdir():
        bboxes[file.stem] = np.load(file)
        
    last_image = ""
    with open("data/frames/frame_manifest.csv", newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            image, image_id, height, width, video_id = row[:5]
            
            if image != last_image:
                last_image = image
                counter = 0
            
            entry = {
                "image": image,
                "image_id": image_id,
                "height": height,
                "width": width,
                "objects": {
                    "id": [], # TODO
                    "area": [], # TODO
                    "bbox": [
                        list(bboxes[video_id][counter]),
                    ],
                    "category": [], # TODO
                },
            }
            
            counter += 1
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
            "category": Sequence(Value("int")),
        },
    })
    
    dataset = Dataset.from_list(rows, features=features)
    dataset.save_to_disk(path)
            
def main():
    build_dataset("data/dataset")
            
if __name__ == "__main__":
    main()
