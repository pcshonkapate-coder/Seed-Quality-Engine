<div align="center">

# 🌱 SeedVision

### Because Every Seed Matters: AI-Powered Quality & Defect Detection

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=data:image/svg+xml;base64,)](https://ultralytics.com)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-Roboflow-purple?logo=data:image/png;base64,)](https://roboflow.com)
[![License](https://img.shields.io/badge/License-Private-red)](.)

</div>

---

Welcome to **SeedVision**! This project uses computer vision to automatically inspect seeds, identifying defects and classifying their overall quality in real time. We leverage the power of YOLOv8 to make seed sorting faster, smarter, and way more accurate.

> [!NOTE]
> **What This Is**
> SeedVision is a deep learning system trained on 1,494 annotated images to detect 12 different conditions—everything from perfectly healthy seeds to specific damage types like cracks, holes, mold, and insect marks.

---

## 🏆 How Good Is It?

We put the model to the test on a separate batch of 150 images it had never seen before. Here's how it did:

> [!TIP]
> **The Big Picture**
> - **Accuracy (mAP@50):** 81.4%
> - **Precision (How often was it right?):** 77.9%
> - **Recall (How much did it find?):** 82.0%
> - **F1-Score (The balance):** 76.2%

### 🔬 The Details (Per-Class)

Some defects are super easy to spot, while others are a bit trickier. Here's the breakdown:

| Condition | Precision | Recall | F1-Score | Average Precision | What it means |
|:---|:---:|:---:|:---:|:---:|:---|
| `damage` | 100.0% | 100.0% | **100.0%** | 99.5% 🥇 | Flawless detection! |
| `crack` | 83.1% | 100.0% | **90.8%** | 99.5% 🥇 | Finds every crack, maybe a few false alarms. |
| `insectdamage` | 100.0% | 66.7% | **80.0%** | 83.3% ✅ | Very sure when it finds insect marks. |
| `healthy-70-60` | 73.7% | 86.2% | **79.4%** | 85.5% ✅ | Solid performance. |
| `hole` | 100.0% | 60.0% | **75.0%** | 80.0% ✅ | Never wrong, but misses a few. |
| `molddamage` | 87.5% | 58.3% | **70.0%** | 73.5% ✅ | Mostly accurate when it spots mold. |
| `healthy` | 66.7% | 66.7% | **66.7%** | 66.7% 🟡 | Needs a bit more data to be certain. |
| `black_point` | 50.0% | 100.0% | **66.7%** | 49.7% 🟡 | Finds them all, but struggles with false positives. |

---

## 🧠 Under the Hood

> [!IMPORTANT]
> **The Tech Specs**
> We're using **YOLOv8s** (the "Small" variant). It hits the sweet spot:
> - **Size:** Just 11.1 Million parameters.
> - **Speed:** Lightning fast. ~94ms per image on a GPU, and still a very respectable ~111ms on a standard laptop CPU.
> - **Head Start:** We didn't start from scratch; we fine-tuned weights pre-trained on the massive COCO dataset.

---

## 📁 The Data Collection

> [!NOTE]
> **Where did the data come from?**
> The dataset was curated and exported from Roboflow in March 2026.
> - **Total Images:** 1,494
> - **How we split it up:**
>   - Train: 1,195 (80%)
>   - Validation: 149 (10%)
>   - Test: 150 (10%)

---

## 🚀 Get Started Now

Ready to try it out? It's straightforward.

> [!TIP]
> **1. Run a Prediction!**
> Want to check an image right now?
> Open `predict.py`, change line 18 (`IMAGE_PATH = "..."`) to point to your image, and run:
> ```bash
> cd SeedVision-master
> ./venv/bin/python predict.py
> ```
> We'll print out a clean verdict and save an annotated image to `predictions/output/`.

> [!NOTE]
> **2. Check the Charts**
> Curious about the graphs? Run the full evaluation on the test set:
> ```bash
> ./venv/bin/python evaluate.py
> ```
> This generates confusion matrices, PR curves, and more, saving them straight into the `evaluation/report/` folder.

> [!IMPORTANT]
> **3. Train Some More**
> Think the model needs to learn a bit more? Just launch the training script. It handles GPU/CPU detection automatically:
> ```bash
> ./venv/bin/python train.py
> ```

---

<div align="center">

**Built with ❤️ using [Ultralytics YOLOv8](https://ultralytics.com)**

</div>
