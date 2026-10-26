import os
import glob
import random
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
SEGMENTED_DIR = r"datasets\segmented\images"
BACKGROUND_DIR = r"datasets\backgrounds"
OUTPUT_DIR = r"datasets\synthetic"

OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_LBL_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

# Dataset Class Mapping (Map subfolder names in datasets/segmented/images to integer IDs)
# Update this dictionary if your class folder names differ!
# Dataset Class Mapping (Map subfolder names in datasets/segmented/images to integer IDs)
# Note: Potato folders use 3 underscores ("Potato___..."), Tomato folders use single underscore ("Tomato_...")
CLASS_MAP = {
    "Potato___Early_blight": 0,
    "Potato___Late_blight": 1,
    "Potato___healthy": 2,
    "Tomato_Early_blight": 3,
    "Tomato_Late_blight": 4,
    "Tomato_healthy": 5
}

# Generation Parameters
NUM_SYNTHETIC_IMAGES = 1200   # 1,200 images for clean 80/10/10 split (960 train, 120 val, 120 test)
CANVAS_SIZE = (640, 640)      # Target size matching YOLO / RT-DETR standard input

# GSD & Rotation Parameters for ~5m UAV Flight Altitude
# Realistic leaf lengths (8 - 18 cm) scaled to ~28px to ~74px (approx 0.045 to 0.115 of canvas)
SCALE_RANGE = (0.045, 0.115)  # Scale relative to canvas size
MAX_OVERLAP_IOU = 0.50        # Allows tight natural clustering while keeping bounding boxes valid
SHADOW_OPACITY = 0.30         # Opacity of realistic drop shadows
SHADOW_OFFSET = (2, 3)        # Scaled shadow offset for small leaves
SHADOW_BLUR = 2               # Blur radius for smaller leaf shadows

def sample_canopy_leaves_count():
    """
    Samples leaf count following realistic multi-stage agricultural field distribution:
    - 20% Emerging/young crops or edge rows: 15 to 28 leaves
    - 60% Standard mature canopy: 35 to 65 leaves
    - 20% Dense closed canopy: 66 to 95 leaves
    """
    r = random.random()
    if r < 0.20:
        return random.randint(15, 28)
    elif r < 0.80:
        return random.randint(35, 65)
    else:
        return random.randint(66, 95)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_all_backgrounds():
    """Gathers all background tiles from datasets/backgrounds (all subfolders)."""
    valid_exts = (".jpg", ".jpeg", ".png", ".tif")
    bg_files = glob.glob(os.path.join(BACKGROUND_DIR, "**", "*.*"), recursive=True)
    return [f for f in bg_files if f.lower().endswith(valid_exts)]

def get_segmented_leaves():
    """Maps class IDs to list of filepaths for RGBA leaf cutouts."""
    leaf_dict = {}
    for class_name, class_id in CLASS_MAP.items():
        folder_path = os.path.join(SEGMENTED_DIR, class_name)
        if os.path.exists(folder_path):
            leaves = glob.glob(os.path.join(folder_path, "*.png"))
            if leaves:
                leaf_dict[class_id] = leaves
    return leaf_dict

def generate_drop_shadow(alpha_channel, offset=SHADOW_OFFSET, blur=SHADOW_BLUR, opacity=SHADOW_OPACITY):
    """Generates a soft, directional drop shadow from an alpha mask."""
    # Expand shadow canvas so offset and blur are not clipped at image boundaries
    pad_w = offset[0] + blur * 2
    pad_h = offset[1] + blur * 2
    shadow_w = alpha_channel.width + max(0, pad_w)
    shadow_h = alpha_channel.height + max(0, pad_h)
    
    shadow = Image.new("L", (shadow_w, shadow_h), 0)
    shadow_alpha = alpha_channel.point(lambda p: int(p * opacity) if p > 0 else 0)
    shadow.paste(shadow_alpha, offset, mask=shadow_alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    return shadow

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0
    return interArea / float(unionArea)

def get_tight_bbox(rotated_rgba):
    """Finds the tight bounding box of non-transparent pixels in an RGBA image."""
    alpha = np.array(rotated_rgba.split()[-1])
    non_zero = np.argwhere(alpha > 20)  # Threshold noise
    if len(non_zero) == 0:
        return None
    y_min, x_min = non_zero.min(axis=0)
    y_max, x_max = non_zero.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)

# -----------------------------------------------------------------------------
# Main Generation Pipeline
# -----------------------------------------------------------------------------
def generate_synthetic_dataset():
    background_paths = get_all_backgrounds()
    leaf_dict = get_segmented_leaves()

    if not background_paths:
        raise FileNotFoundError(f"No background images found in '{BACKGROUND_DIR}'. Check your folders!")
    if not leaf_dict:
        raise FileNotFoundError(f"No RGBA leaf cutouts found in '{SEGMENTED_DIR}'. Check your folder names and CLASS_MAP!")

    available_classes = list(leaf_dict.keys())
    print(f"Loaded {len(background_paths)} background base plates.")
    print(f"Available classes for generation: {[(name, CLASS_MAP[name], len(leaf_dict[CLASS_MAP[name]])) for name in CLASS_MAP if CLASS_MAP[name] in leaf_dict]}")
    print(f"Generating {NUM_SYNTHETIC_IMAGES} composite images...")

    for img_idx in range(1, NUM_SYNTHETIC_IMAGES + 1):
        # 1. Select and prepare canvas background
        bg_path = random.choice(background_paths)
        canvas = Image.open(bg_path).convert("RGB")
        if canvas.size != CANVAS_SIZE:
            canvas = canvas.resize(CANVAS_SIZE, Image.Resampling.BILINEAR)

        # Apply mild global lighting variation to background
        brightness_factor = random.uniform(0.85, 1.15)
        canvas = ImageEnhance.Brightness(canvas).enhance(brightness_factor)

        existing_boxes = []  # Store [x1, y1, x2, y2] to track overlaps
        yolo_labels = []     # Store formatted YOLO lines

        num_leaves = sample_canopy_leaves_count()

        # Generate 2 to 5 natural plant cluster center points per frame (mimicking crop bushes / plant beds)
        num_clusters = random.randint(2, 5)
        cluster_centers = [
            (random.randint(80, CANVAS_SIZE[0] - 80), random.randint(80, CANVAS_SIZE[1] - 80))
            for _ in range(num_clusters)
        ]

        for _ in range(num_leaves):
            # 2. Pick a random class and leaf cutout
            class_id = random.choice(available_classes)
            leaf_path = random.choice(leaf_dict[class_id])
            leaf_img = Image.open(leaf_path).convert("RGBA")

            # 3. Apply Domain Randomization (GSD scaling, Rotation, Color Jitter)
            scale = random.uniform(*SCALE_RANGE)
            target_w = max(14, int(CANVAS_SIZE[0] * scale))
            aspect_ratio = leaf_img.height / max(1, leaf_img.width)
            target_h = max(14, int(target_w * aspect_ratio))

            leaf_resized = leaf_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
            # Subtle lighting/color randomization on leaf to blend with background
            rgb_leaf = leaf_resized.convert("RGB")
            alpha_ch = leaf_resized.split()[-1]
            rgb_leaf = ImageEnhance.Brightness(rgb_leaf).enhance(random.uniform(0.85, 1.20))
            rgb_leaf = ImageEnhance.Color(rgb_leaf).enhance(random.uniform(0.80, 1.20))
            leaf_resized = rgb_leaf.convert("RGBA")
            leaf_resized.putalpha(alpha_ch)

            angle = random.uniform(0, 360)
            leaf_rotated = leaf_resized.rotate(angle, expand=True, resample=Image.Resampling.BILINEAR)

            # Feather edges slightly to remove SAM sharpness ("sticker effect")
            r, g, b, a = leaf_rotated.split()
            a_feathered = a.filter(ImageFilter.GaussianBlur(0.8))
            leaf_rotated.putalpha(a_feathered)

            # 4. Determine placement coordinates clustered around plant centers
            leaf_w, leaf_h = leaf_rotated.size
            if leaf_w >= CANVAS_SIZE[0] or leaf_h >= CANVAS_SIZE[1]:
                continue

            max_x = CANVAS_SIZE[0] - leaf_w
            max_y = CANVAS_SIZE[1] - leaf_h

            # Cluster distribution: 80% around cluster centers (Gaussian spread), 20% uniformly scattered
            if random.random() < 0.80:
                cx, cy = random.choice(cluster_centers)
                spread = random.uniform(40, 95)  # Natural plant canopy radius
                pos_x = int(random.gauss(cx - leaf_w // 2, spread))
                pos_y = int(random.gauss(cy - leaf_h // 2, spread))
                pos_x = max(0, min(pos_x, max_x))
                pos_y = max(0, min(pos_y, max_y))
            else:
                pos_x = random.randint(0, max_x)
                pos_y = random.randint(0, max_y)

            # 5. Extract precise bounding box around visible leaf pixels
            tight_bounds = get_tight_bbox(leaf_rotated)
            if tight_bounds is None:
                continue

            tb_x1, tb_y1, tb_x2, tb_y2 = tight_bounds
            
            box_x1 = pos_x + tb_x1
            box_y1 = pos_y + tb_y1
            box_x2 = pos_x + tb_x2
            box_y2 = pos_y + tb_y2

            # Check for excessive occlusion against previously placed leaves
            candidate_box = [box_x1, box_y1, box_x2, box_y2]
            too_much_overlap = any(calculate_iou(candidate_box, b) > MAX_OVERLAP_IOU for b in existing_boxes)
            if too_much_overlap:
                continue

            # 6. Apply Grounding Drop Shadow onto canvas
            shadow_img = generate_drop_shadow(leaf_rotated.split()[-1])
            shadow_layer = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
            shadow_layer.paste(shadow_img, (pos_x, pos_y), mask=shadow_img)
            
            # Blend shadow with background
            canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow_layer).convert("RGB")

            # 7. Layer leaf cutout onto canvas
            canvas.paste(leaf_rotated, (pos_x, pos_y), mask=leaf_rotated)
            existing_boxes.append(candidate_box)

            # 8. Convert BBOX to Normalized YOLO format (x_center, y_center, width, height)
            bbox_w = max(1, box_x2 - box_x1)
            bbox_h = max(1, box_y2 - box_y1)
            x_center = (box_x1 + bbox_w / 2.0) / CANVAS_SIZE[0]
            y_center = (box_y1 + bbox_h / 2.0) / CANVAS_SIZE[1]
            norm_w = bbox_w / CANVAS_SIZE[0]
            norm_h = bbox_h / CANVAS_SIZE[1]

            # Clip normalized coordinates within [0.0, 1.0]
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            norm_w = min(max(norm_w, 0.0), 1.0)
            norm_h = min(max(norm_h, 0.0), 1.0)

            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        # Optional subtle UAV optical degradation: slight camera blur or grain
        if random.random() < 0.25:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 0.9)))

        # 9. Save image and matching label file
        file_base = f"synth_{img_idx:05d}"
        img_output_path = os.path.join(OUTPUT_IMG_DIR, f"{file_base}.jpg")
        lbl_output_path = os.path.join(OUTPUT_LBL_DIR, f"{file_base}.txt")

        canvas.save(img_output_path, "JPEG", quality=95)
        with open(lbl_output_path, "w") as f:
            f.write("\n".join(yolo_labels))

        if img_idx % 100 == 0 or img_idx == NUM_SYNTHETIC_IMAGES:
            print(f"Generated [{img_idx}/{NUM_SYNTHETIC_IMAGES}] synthetic samples...")

    print(f"\nSynthetic generation complete!")
    print(f"Images saved to: {OUTPUT_IMG_DIR}")
    print(f"Labels saved to: {OUTPUT_LBL_DIR}")

if __name__ == "__main__":
    generate_synthetic_dataset()