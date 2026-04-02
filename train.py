"""
train.py — SeedVision YOLOv8 Training Script
Run with the project venv:
    ./venv/bin/python train.py
"""

from ultralytics import YOLO
import os
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE, "data.yaml")

# Auto-detect GPU or fall back to CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("  SeedVision — YOLOv8s Training")
print(f"  Device      : {'GPU (CUDA)' if DEVICE == 0 else 'CPU'}")
print(f"  Data config : {DATA_YAML}")
print("=" * 60)

# Load pretrained YOLOv8s (downloads COCO weights automatically on first run)
model = YOLO("yolov8s.pt")

results = model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=16 if DEVICE == 0 else 8,   # smaller batch on CPU
    patience=20,                        # early stop if no improvement
    device=DEVICE,
    workers=4,
    name="seedvision_v1",
    project=os.path.join(BASE, "runs"),
    exist_ok=True,
    # Augmentation — very helpful for small datasets
    mosaic=1.0,
    mixup=0.1,
    degrees=10.0,
    flipud=0.1,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    # Freeze backbone for first 10 epochs (transfer learning)
    freeze=10,
)

print("\n Training complete!")
best = os.path.join(BASE, "runs", "seedvision_v1", "weights", "best.pt")
print(f"   Best weights : {best}")
print(f"   mAP@50       : {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
