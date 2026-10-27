from pathlib import Path
from ultralytics import YOLO, RTDETR

DATA_REAL_YAML = str(Path("data_real.yaml").resolve())

# Point to your existing synthetic checkpoints
YOLO_SYNTHETIC = Path("runs/detect/runs/train/yolo11m_uav_disease-4/weights/best.pt")
RTDETR_SYNTHETIC = Path("runs/detect/runs/train/rtdetr_l_uav_disease-2/weights/best.pt")

def finetune():
    # 1. Fine-tune YOLOv11m
    if YOLO_SYNTHETIC.exists():
        print("--> Fine-tuning YOLOv11m on Real UAV Dataset...")
        model_yolo = YOLO(str(YOLO_SYNTHETIC))
        model_yolo.train(
            data=DATA_REAL_YAML,
            epochs=30,
            imgsz=256,  # Zenodo images are 256x256 tiles
            batch=16,
            project="runs/train_real",
            name="yolo11m_real_uav",
            lr0=0.0001, # Lower learning rate to preserve synthetic pre-trained features
            optimizer="AdamW"
        )
    else:
        print(f"[-] YOLO synthetic weights not found at: {YOLO_SYNTHETIC}")

    # 2. Fine-tune RT-DETR-L
    if RTDETR_SYNTHETIC.exists():
        print("--> Fine-tuning RT-DETR-L on Real UAV Dataset...")
        model_rtdetr = RTDETR(str(RTDETR_SYNTHETIC))
        model_rtdetr.train(
            data=DATA_REAL_YAML,
            epochs=30,
            imgsz=256,
            batch=8,
            project="runs/train_real",
            name="rtdetr_real_uav",
            lr0=0.00005,
            optimizer="AdamW"
        )
    else:
        print(f"[-] RT-DETR synthetic weights not found at: {RTDETR_SYNTHETIC}")

if __name__ == "__main__":
    finetune()
