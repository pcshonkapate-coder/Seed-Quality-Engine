"""
predict.py — SeedVision Inference
==================================
1. Set IMAGE_PATH to your image file below
2. Run: python predict.py   (works from any terminal or IDE)
"""

import sys, os

# Auto-load venv so this works with system python too
_venv_site = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "venv", "lib")
if os.path.isdir(_venv_site):
    for _entry in os.listdir(_venv_site):  # e.g. python3.12
        _sp = os.path.join(_venv_site, _entry, "site-packages")
        if os.path.isdir(_sp) and _sp not in sys.path:
            sys.path.insert(0, _sp)

from ultralytics import YOLO

# ──────────────────────────────────────────
#    CHANGE THIS to your image path
IMAGE_PATH = "/home/shon/SeedVisiion/test.png"
# ──────────────────────────────────────────

CONFIDENCE  = 0.30
BASE        = os.path.dirname(os.path.abspath(__file__))
WEIGHTS     = os.path.join(BASE, "runs", "seedvision_v1", "weights", "best.pt")

BAD_CLASSES = {"crack", "damage", "hole", "insectdamage",
               "molddamage", "black_point", "shriveledseed", "staindamage"}

model   = YOLO(WEIGHTS)
results = model.predict(IMAGE_PATH, conf=CONFIDENCE, verbose=False, save=True,
                        project=os.path.join(BASE, "predictions"), name="output",
                        exist_ok=True)

result = results[0]
boxes  = result.boxes

print("\n" + "=" * 50)
print(f"  SeedVision — {os.path.basename(IMAGE_PATH)}")
print("=" * 50)

if len(boxes) == 0:
    print("    No seeds detected. Try lowering CONFIDENCE.")
else:
    for box in boxes:
        cls_id      = int(box.cls[0])
        name        = model.names[cls_id]
        conf_v      = float(box.conf[0])
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        status      = " DEFECTIVE" if name in BAD_CLASSES else " HEALTHY"
        print(f"  {status}  |  {name:<18}  conf={conf_v:.0%}  bbox=[{x1},{y1},{x2},{y2}]")

    n_bad = sum(1 for b in boxes if model.names[int(b.cls)] in BAD_CLASSES)
    print("-" * 50)
    print(f"  Total: {len(boxes)}  |  Healthy: {len(boxes)-n_bad}  |  Defective: {n_bad}")
    print(f"  Verdict: {' DEFECT(S) FOUND' if n_bad else ' ALL HEALTHY'}")

saved = os.path.join(BASE, "predictions", "output", os.path.basename(IMAGE_PATH))
print(f"\n  Annotated image → {saved}")
print("=" * 50 + "\n")
