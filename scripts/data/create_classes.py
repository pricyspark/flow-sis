import csv
import json
import os
from pathlib import Path
from collections.abc import Iterable
from typing import Dict, List, Tuple

def create_json(
    csv_path: Path, 
    json_path: Path, 
    class_cols: Iterable[int | str]
) -> None:
    output: Tuple[Dict[str, str], Dict[str, int], List[str]] = ({}, {}, [])
    vid2label: Dict[str, str] = {}
    label2id: Dict[str, int] = {}
    id2label: List[str] = []
    output = (vid2label, label2id, id2label)
    with open(csv_path, newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            components = [
                row[col] 
                if isinstance(col, int)
                else row[header.index(col)]
                for col in class_cols]
            c = " ".join(components)
            vid2label[row[0]] = c
            if c not in label2id:
                label2id[c] = len(label2id)
                id2label.append(c)
            
    with open(json_path, 'w') as file:
        json.dump(output, file, indent=2)
        
def main():
    create_json(
        Path("data/raw/video_manifest.csv"),
        Path("data/raw/classes.json"),
        ["ToolStyle", "Class"],
    )
    
if __name__ == "__main__":
    main()
   
    
'''
Perkins 113

Dougherty 331
'''