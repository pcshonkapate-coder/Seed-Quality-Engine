"""
split_dataset.py
Splits the existing train/ images+labels into train / valid / test sets.
Run: venv/bin/python split_dataset.py
"""

import os
import shutil
import random

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_IMAGES = os.path.join(BASE, "train", "images")
SRC_LABELS = os.path.join(BASE, "train", "labels")

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
# TEST_RATIO  = 0.10  (remainder)

random.seed(42)

all_images = [f for f in os.listdir(SRC_IMAGES) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_images)

n = len(all_images)
n_train = int(n * TRAIN_RATIO)
n_valid = int(n * VALID_RATIO)

splits = {
    "train": all_images[:n_train],
    "valid": all_images[n_train:n_train + n_valid],
    "test":  all_images[n_train + n_valid:]
}

print(f"Total images: {n}")
for split_name, imgs in splits.items():
    print(f"  {split_name}: {len(imgs)} images")

# Create dirs
for split_name in splits:
    for subdir in ("images", "labels"):
        os.makedirs(os.path.join(BASE, split_name, subdir), exist_ok=True)

# Copy files — skip if source and destination are the same file
for split_name, imgs in splits.items():
    for img_file in imgs:
        stem = os.path.splitext(img_file)[0]
        label_file = stem + ".txt"

        src_img = os.path.join(SRC_IMAGES, img_file)
        dst_img = os.path.join(BASE, split_name, "images", img_file)
        if os.path.abspath(src_img) != os.path.abspath(dst_img):
            shutil.copy2(src_img, dst_img)

        src_lbl = os.path.join(SRC_LABELS, label_file)
        dst_lbl = os.path.join(BASE, split_name, "labels", label_file)
        if os.path.exists(src_lbl):
            if os.path.abspath(src_lbl) != os.path.abspath(dst_lbl):
                shutil.copy2(src_lbl, dst_lbl)
        else:
            print(f"  [WARN] No label for {img_file}")

print("\nDone! Dataset split complete.")
