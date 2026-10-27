import os
import shutil
import random
from pathlib import Path

# Paths
ZENODO_DIR = Path(r"C:\Users\agroeye\Downloads\Alternaria_2022\2022")
REAL_DATASET_DIR = Path("datasets/real_zenodo")

# Class map: 0 = Potato Early Blight (Alternaria solani)
CLASS_ID = 0  

def build_real_dataset():
    random.seed(42)

    # 1. Create independent directory tree
    for split in ["train", "val", "test"]:
        (REAL_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (REAL_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. Gather infected patches (Folder 1)
    folder_1 = ZENODO_DIR / "1"
    infected_images = list(folder_1.glob("*.jpg")) + list(folder_1.glob("*.png")) + list(folder_1.glob("*.bmp"))
    
    # 3. Gather healthy/background patches (Folder 0)
    folder_0 = ZENODO_DIR / "0"
    background_images = list(folder_0.glob("*.jpg")) + list(folder_0.glob("*.png")) + list(folder_0.glob("*.bmp"))

    print(f"Found {len(infected_images)} infected images and {len(background_images)} background images.")

    random.shuffle(infected_images)
    random.shuffle(background_images)

    # 80/10/10 Split ratios
    n_inf = len(infected_images)
    splits_inf = {
        "train": infected_images[:int(n_inf * 0.8)],
        "val": infected_images[int(n_inf * 0.8):int(n_inf * 0.9)],
        "test": infected_images[int(n_inf * 0.9):]
    }

    # Process infected patches (with center bounding box)
    for split_name, image_list in splits_inf.items():
        img_dest = REAL_DATASET_DIR / "images" / split_name
        lbl_dest = REAL_DATASET_DIR / "labels" / split_name

        for img_path in image_list:
            # Copy image
            shutil.copy2(img_path, img_dest / img_path.name)

            # Create bounding box label covering the central patch lesion
            lbl_path = lbl_dest / (img_path.stem + ".txt")
            with open(lbl_path, "w") as f:
                f.write(f"{CLASS_ID} 0.500000 0.500000 0.700000 0.700000\n")

    # Process background patches (no label file = hard negative samples)
    n_bg = len(background_images)
    splits_bg = {
        "train": background_images[:int(n_bg * 0.8)],
        "val": background_images[int(n_bg * 0.8):int(n_bg * 0.9)],
        "test": background_images[int(n_bg * 0.9):]
    }

    for split_name, image_list in splits_bg.items():
        img_dest = REAL_DATASET_DIR / "images" / split_name
        for img_path in image_list:
            shutil.copy2(img_path, img_dest / img_path.name)
            # Intentionally no label file generated so YOLO treats it as negative background

    print(f"\n[+] Real UAV dataset populated at: {REAL_DATASET_DIR.resolve()}")

if __name__ == "__main__":
    build_real_dataset()
