import os
from pathlib import Path


def delete_dir(path: Path):
    files = []
    for file in path.iterdir():
        if not file.is_file():
            continue
        
        if file.stem == ".gitkeep":
            continue
        
        files.append(file)
            
    for file in files:
        os.remove(file)
        
        
def confirm_delete():
    response = input(
        "About to delete all generated data files.\n"
        "Type 'yes' to continue: "
    )
    
    return response.strip().lower() == "yes"
        
        
def main():
    if confirm_delete():
        delete_dir(Path("data/frames"))
        delete_dir(Path("data/boxes"))
        delete_dir(Path("data/masks"))
        delete_dir(Path("data/dataset"))
        print("Deleted")
    else:
        print("Aborted")
        
        
if __name__ == "__main__":
    main()