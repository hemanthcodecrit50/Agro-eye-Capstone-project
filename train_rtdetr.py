"""
Train RT-DETR-L on the UAV Potato & Tomato Disease dataset.
Run: .\\venv\\Scripts\\python.exe train_rtdetr.py
Results saved to: runs/train/rtdetr_l_uav_disease/
"""
from pathlib import Path
from ultralytics import RTDETR
import torch
from tqdm import tqdm

DATA_YAML = str(Path("data.yaml").resolve())
EPOCHS = 40


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    print("Training RT-DETR-L  |  Vision Transformer (NMS-Free)")
    print("=" * 60)

    model = RTDETR("rtdetr-l.pt")

    # --- Epoch-level tqdm progress bar via Ultralytics callbacks ---
    epoch_bar = tqdm(total=EPOCHS, desc="RT-DETR-L Epochs", unit="epoch",
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
        batch=8,
        device=device,
        project="runs/train",
        name="rtdetr_l_uav_disease",
        save=True,
        plots=True,
        workers=4,
        patience=15,
        optimizer="AdamW",
        lr0=0.0001,
        verbose=True
    )

    print("\nRT-DETR-L training complete!")
    print("Results saved to: runs/train/rtdetr_l_uav_disease/")


if __name__ == "__main__":
    main()
