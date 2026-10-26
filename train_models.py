import sys
from pathlib import Path
from ultralytics import YOLO, RTDETR
import torch
from tqdm import tqdm

DATA_YAML = str(Path("data.yaml").resolve())

def train_models():
    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Using Device: {device} ({device_name})\n")

    models_to_train = [
        {
            "name": "YOLOv11m (CNN SOTA Baseline)",
            "model_cls": YOLO,
            "weights": "yolo11m.pt",
            "run_name": "yolo11m_uav_disease",
            "batch": 16,
            "lr0": 0.001,
            "epochs": 40
        },
        {
            "name": "RT-DETR-L (Vision Transformer Baseline)",
            "model_cls": RTDETR,
            "weights": "rtdetr-l.pt",
            "run_name": "rtdetr_l_uav_disease",
            "batch": 8,
            "lr0": 0.0001,
            "epochs": 40
        }
    ]

    print("="*65)
    print("STARTING UAV DISEASE DETECTION MODEL TRAINING")
    print("="*65)

    # Master progress bar across models
    with tqdm(models_to_train, desc="Models Pipeline", unit="model") as pbar:
        for cfg in pbar:
            pbar.set_postfix({"Current": cfg["name"]})
            print(f"\n---> Launching: {cfg['name']}")

            model = cfg["model_cls"](cfg["weights"])
            
            # Ultralytics natively manages an internal tqdm progress bar per epoch and batch
            results = model.train(
                data=DATA_YAML,
                epochs=cfg["epochs"],
                imgsz=640,
                batch=cfg["batch"],
                device=device,
                project="runs/train",
                name=cfg["run_name"],
                save=True,
                plots=True,
                workers=4,
                patience=15,
                optimizer="AdamW",
                lr0=cfg["lr0"]
            )
            print(f"---> Finished training {cfg['name']}! Artifacts saved to runs/train/{cfg['run_name']}\n")

    print("\n" + "="*65)
    print("ALL MODEL TRAININGS COMPLETED SUCCESSFULLY!")
    print("="*65)

if __name__ == "__main__":
    train_models()
