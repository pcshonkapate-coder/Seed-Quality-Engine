import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import tempfile

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SeedVision",
    page_icon="",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0e1117; }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4ade80, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .detection-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .conf-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #4ade80, #22d3ee);
        margin-top: 4px;
    }
    .status-box {
        background: #1e293b;
        border-left: 4px solid #4ade80;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        color: #94a3b8;
        font-size: 0.85rem;
    }
    .stFileUploader > div { border: 2px dashed #334155 !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
WEIGHTS     = os.path.join(BASE, "runs", "seedvision_v1", "weights", "best.pt")

CLASS_EMOJI = {
    "healthy":      "",
    "healthy-50-40":"",
    "healthy-70-60":"",
    "crack":        "",
    "damage":       "",
    "hole":         "",
    "insectdamage": "",
    "molddamage":   "",
    "black_point":  "",
    "shriveledseed":"",
    "staindamage":  "",
    "objects":      "",
}

BAD_CLASSES = {"crack","damage","hole","insectdamage","molddamage",
               "black_point","shriveledseed","staindamage"}

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(WEIGHTS):
        st.error(" Model weights not found. Training may still be in progress.")
        st.stop()
    return YOLO(WEIGHTS)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title"> SeedVision</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered seed quality & defect detection · YOLOv8</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Settings")
    conf    = st.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05)
    iou     = st.slider("NMS IoU threshold",    0.10, 0.90, 0.45, 0.05)
    show_bb = st.checkbox("Show bounding boxes", value=True)
    st.markdown("---")
    st.markdown("###  Classes")
    for name, emoji in CLASS_EMOJI.items():
        color = "" if name in BAD_CLASSES else ""
        st.markdown(f"{emoji} `{name}` {color}")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a seed image",
    type=["jpg","jpeg","png","bmp","webp"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div class="status-box">
     Upload an image above to begin detection.<br>
    Supports JPG, PNG, BMP, WEBP.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Run inference ─────────────────────────────────────────────────────────────
with st.spinner(" Analysing seed..."):
    model = load_model()

    img_pil  = Image.open(uploaded).convert("RGB")
    img_np   = np.array(img_pil)

    # Save to temp file (YOLO needs a path)
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1],
                                     delete=False) as tmp:
        img_pil.save(tmp.name)
        results = model.predict(tmp.name, conf=conf, iou=iou, verbose=False)
    os.unlink(tmp.name)

    result = results[0]

# ── Display images ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Original**")
    st.image(img_pil, use_container_width=True)

with col2:
    st.markdown("**Detections**")
    if show_bb:
        annotated = result.plot()   # BGR numpy
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated, use_container_width=True)
    else:
        st.image(img_pil, use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("---")
boxes = result.boxes

if len(boxes) == 0:
    st.warning(" No seeds detected. Try lowering the confidence threshold.")
else:
    n_defect  = sum(1 for b in boxes if model.names[int(b.cls)] in BAD_CLASSES)
    n_healthy = len(boxes) - n_defect
    verdict   = " GOOD" if n_defect == 0 else f" {n_defect} DEFECT(S) FOUND"

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Detected", len(boxes))
    m2.metric("Healthy",  n_healthy)
    m3.metric("Defective", n_defect)

    st.markdown(f"### Verdict: **{verdict}**")
    st.markdown("#### Detections")

    for box in boxes:
        cls_id  = int(box.cls[0])
        name    = model.names[cls_id]
        conf_v  = float(box.conf[0])
        emoji   = CLASS_EMOJI.get(name, "")
        bar_w   = int(conf_v * 100)
        status  = " Defective" if name in BAD_CLASSES else " Healthy"

        st.markdown(f"""
        <div class="detection-card">
            <span style="font-size:1.5rem">{emoji}</span>
            <div style="flex:1">
                <div style="display:flex;justify-content:space-between">
                    <strong style="color:#e2e8f0">{name}</strong>
                    <span style="color:#94a3b8">{status}</span>
                </div>
                <div style="display:flex;align-items:center;gap:8px;margin-top:4px">
                    <div class="conf-bar" style="width:{bar_w}%"></div>
                    <span style="color:#94a3b8;font-size:0.8rem">{conf_v:.0%}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Download annotated image ──────────────────────────────────────────────────
    if show_bb:
        import io
        annotated_pil = Image.fromarray(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        st.download_button(
            label=" Download annotated image",
            data=buf.getvalue(),
            file_name=f"seedvision_{uploaded.name}",
            mime="image/png",
            use_container_width=True,
        )

st.markdown("---")
st.markdown('<p style="text-align:center;color:#475569;font-size:0.8rem">SeedVision · YOLOv8s · Trained on 1,195 seed images · 12 classes</p>', unsafe_allow_html=True)
