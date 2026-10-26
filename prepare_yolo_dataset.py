import os
import shutil
import random
from pathlib import Path

# Paths
SOURCE_IMG_DIR = Path("datasets/synthetic/images")
SOURCE_LBL_DIR = Path("datasets/synthetic/labels")
YOLO_DIR = Path("datasets/yolo_dataset")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

def prepare_yolo_dataset():
    random.seed(42)
    
    # Target folders
    for split in ["train", "val", "test"]:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(SOURCE_IMG_DIR.glob("*.jpg")))
    print(f"Total synthetic images to partition: {len(img_files)}")
    
    valid_pairs = []
    for img_path in img_files:
        lbl_path = SOURCE_LBL_DIR / (img_path.stem + ".txt")
        if lbl_path.exists():
            valid_pairs.append((img_path, lbl_path))

    random.shuffle(valid_pairs)

    n_total = len(valid_pairs)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    splits = {
        "train": valid_pairs[:n_train],
        "val": valid_pairs[n_train:n_train + n_val],
        "test": valid_pairs[n_train + n_val:]
    }

    for split_name, pairs in splits.items():
        img_dest = YOLO_DIR / "images" / split_name
        lbl_dest = YOLO_DIR / "labels" / split_name
        
        # Clean dest
        for f in img_dest.glob("*.*"):
            f.unlink()
        for f in lbl_dest.glob("*.*"):
            f.unlink()

        for img_p, lbl_p in pairs:
            shutil.copy2(img_p, img_dest / img_p.name)
            shutil.copy2(lbl_p, lbl_dest / lbl_p.name)
        
        print(f"[{split_name.upper()}] Partitioned: {len(pairs)} image-label pairs.")

    print("\nDataset preparation complete in 'datasets/yolo_dataset/'!")

if __name__ == "__main__":
    prepare_yolo_dataset()
