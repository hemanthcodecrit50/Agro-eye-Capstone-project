"""
Evaluation and Inference script for trained UAV Potato & Tomato Disease models.
Evaluates RT-DETR-L and YOLOv11m on the unseen 'test' split and runs visual predictions.

Usage:
  .\venv\Scripts\python.exe test_models.py --model yolo
  .\venv\Scripts\python.exe test_models.py --model rtdetr
  .\venv\Scripts\python.exe test_models.py --model both
"""

import argparse
from pathlib import Path
from ultralytics import YOLO, RTDETR

DATA_YAML = str(Path("data.yaml").resolve())

# Discovered latest run checkpoints
YOLO_WEIGHTS = Path("runs/detect/runs/train/yolo11m_uav_disease-4/weights/best.pt")
RTDETR_WEIGHTS = Path("runs/detect/runs/train/rtdetr_l_uav_disease-2/weights/best.pt")


def evaluate_model(name: str, model_class, weights_path: Path):
    if not weights_path.exists():
        print(f"[-] Checkpoint not found at: {weights_path}")
        return

    print("\n" + "=" * 70)
    print(f"Evaluating {name} on unseen TEST split")
    print(f"Checkpoint: {weights_path}")
    print("=" * 70)

    model = model_class(str(weights_path))

    # 1. Quantitative Evaluation on TEST split
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        batch=16,
        plots=True,
        project="runs/evaluate",
        name=f"{name.lower().replace('-', '_')}_test_split"
    )

    print(f"\n[+] {name} Test Set Results:")
    print(f"    mAP@50:    {metrics.box.map50:.4f}")
    print(f"    mAP@50-95: {metrics.box.map:.4f}")
    print(f"    Precision: {metrics.box.mp:.4f}")
    print(f"    Recall:    {metrics.box.mr:.4f}")

    # 2. Visual Inference Demo on first 5 test images
    test_img_dir = Path("datasets/yolo_dataset/images/test")
    sample_images = sorted(list(test_img_dir.glob("*.jpg")))[:5]
    if sample_images:
        print(f"\n[+] Running visual inference on {len(sample_images)} test samples...")
        sample_paths = [str(p) for p in sample_images]
        model.predict(
            source=sample_paths,
            save=True,
            conf=0.35,
            iou=0.6,
            project="runs/predict",
            name=f"{name.lower().replace('-', '_')}_predictions",
            imgsz=640
        )
        print(f"[+] Visual prediction images saved to runs/predict/{name.lower().replace('-', '_')}_predictions/")


def main():
    parser = argparse.ArgumentParser(description="Test trained UAV disease detection models")
    parser.add_argument("--model", choices=["yolo", "rtdetr", "both"], default="both",
                        help="Which model to evaluate (default: both)")
    args = parser.parse_args()

    if args.model in ("yolo", "both"):
        evaluate_model("YOLOv11m", YOLO, YOLO_WEIGHTS)

    if args.model in ("rtdetr", "both"):
        evaluate_model("RT-DETR-L", RTDETR, RTDETR_WEIGHTS)


if __name__ == "__main__":
    main()
