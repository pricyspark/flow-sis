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
    output: Tuple[Dict[str, str], Dict[str, int]] = ({}, {})
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
            output[0][row[0]] = c
            if c not in output:
                output[1][c] = len(output)
            
    with open(json_path, 'w') as file:
        json.dump(output, file)
        
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