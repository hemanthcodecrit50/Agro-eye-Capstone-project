"""
Train YOLOv11m on the UAV Potato & Tomato Disease dataset.
Run: .\\venv\\Scripts\\python.exe train_yolo11m.py
Results saved to: runs/train/yolo11m_uav_disease/
"""
from pathlib import Path
from ultralytics import YOLO
import torch
from tqdm import tqdm

DATA_YAML = str(Path("data.yaml").resolve())
EPOCHS = 40


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    print("Training YOLOv11m  |  CNN SOTA Baseline")
    print("=" * 60)

    model = YOLO("yolo11m.pt")

    # --- Epoch-level tqdm progress bar via Ultralytics callbacks ---
    epoch_bar = tqdm(total=EPOCHS, desc="YOLOv11m Epochs", unit="epoch",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                     dynamic_ncols=True)

    def on_train_epoch_end(trainer):
        metrics = trainer.metrics or {}
        map50    = metrics.get("metrics/mAP50(B)",    0.0)
        map5095  = metrics.get("metrics/mAP50-95(B)", 0.0)
        box_loss = metrics.get("train/box_loss",      0.0)
        epoch_bar.set_postfix({
            "mAP50":     f"{map50:.3f}",
            "mAP50-95":  f"{map5095:.3f}",
            "box_loss":  f"{box_loss:.4f}"
        })
        epoch_bar.update(1)

    def on_train_end(trainer):
        epoch_bar.close()

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end",       on_train_end)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=640,
        batch=16,
        device=device,
        project="runs/train",
        name="yolo11m_uav_disease",
        save=True,
        plots=True,
        workers=4,
        patience=15,
        optimizer="AdamW",
        lr0=0.001,
        verbose=True
    )

    print("\nYOLOv11m training complete!")
    print("Results saved to: runs/train/yolo11m_uav_disease/")


if __name__ == "__main__":
    main()
