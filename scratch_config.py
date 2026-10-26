# Configuration for 5m UAV Flight Dataset
NUM_SYNTHETIC_IMAGES = 1200   # 1,200 images for clean 80/10/10 split (960 train, 120 val, 120 test)
CANVAS_SIZE = (640, 640)      # Standard YOLO / RT-DETR input tile size

# GSD & Rotation Parameters for ~5m UAV Flight Altitude
# Leaves/leaflets are scaled to ~28px to ~72px (0.045 to 0.115 of canvas)
SCALE_RANGE = (0.045, 0.115)  
MAX_OVERLAP_IOU = 0.50        # Realistic canopy overlap tolerance
SHADOW_OPACITY = 0.30         # Natural ambient drop shadow opacity
SHADOW_OFFSET = (2, 3)        # Scaled directional shadow offset
SHADOW_BLUR = 2               # Diffuse shadow blur radius

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
