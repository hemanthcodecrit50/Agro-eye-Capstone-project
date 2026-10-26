import os
import glob
import cv2
import numpy as np
from PIL import Image

RAW_DIR = "datasets/raw_backgrounds"
BG_DIR = "datasets/backgrounds"
TILE_SIZE = (640, 640)
STRIDE = 640

# Create destination subdirectories
for cat in ["farm", "soil", "grass"]:
    os.makedirs(os.path.join(BG_DIR, cat), exist_ok=True)

def categorize_tile(cv_img):
    """Categorizes tile based on HSV green pixel percentage."""
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    
    # Define HSV bounds for green foliage
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / (cv_img.shape[0] * cv_img.shape[1])

    if green_ratio < 0.15:
        return "soil"
    elif green_ratio > 0.70:
        return "grass"
    else:
        return "farm"

def process_backgrounds():
    raw_images = glob.glob(os.path.join(RAW_DIR, "**", "*.*"), recursive=True)
    valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    raw_images = [f for f in raw_images if f.lower().endswith(valid_exts)]
    
    counts = {"farm": 0, "soil": 0, "grass": 0}
    print(f"Processing {len(raw_images)} raw background images...")

    for img_path in raw_images:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                base_name = os.path.splitext(os.path.basename(img_path))[0]

                for y in range(0, h - TILE_SIZE[1] + 1, STRIDE):
                    for x in range(0, w - TILE_SIZE[0] + 1, STRIDE):
                        tile = img.crop((x, y, x + TILE_SIZE[0], y + TILE_SIZE[1]))
                        if tile.mode != "RGB":
                            tile = tile.convert("RGB")
                        
                        # Convert PIL to OpenCV for quick HSV evaluation
                        cv_tile = cv2.cvtColor(np.array(tile), cv2.COLOR_RGB2BGR)
                        category = categorize_tile(cv_tile)

                        save_path = os.path.join(
                            BG_DIR, category, f"{base_name}_{x}_{y}.jpg"
                        )
                        tile.save(save_path, "JPEG", quality=95)
                        counts[category] += 1
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

    print("--- Slicing & Categorization Complete ---")
    for cat, count in counts.items():
        print(f"datasets/backgrounds/{cat}: {count} tiles")

if __name__ == "__main__":
    process_backgrounds()