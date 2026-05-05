# ============================================================
#  Chest X-Ray Diagnostic System
#  Author  : Data Science Student Project
#  Stack   : Streamlit · PyTorch · PIL · NumPy
# ============================================================

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path

from xray.ml.model.arch import Net
from xray.constant.training_pipeline import (
    RESIZE,
    NORMALIZE_LIST_1,
    NORMALIZE_LIST_2,
)

# ──────────────────────────────────────────────
# MODEL PATH
# ──────────────────────────────────────────────
MODEL_PATH = Path("notebook/xray_model.pth")

# ──────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Diagnostic System",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# GLOBAL CSS  — clinical dark + electric teal
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;600;700;800&display=swap');

/* ── Palette ── */
:root {
    --bg-deep:     #050a12;
    --bg-mid:      #0b1220;
    --surface:     rgba(255,255,255,0.04);
    --surface-2:   rgba(255,255,255,0.07);
    --border:      rgba(255,255,255,0.08);
    --border-glow: rgba(0,229,195,0.35);
    --teal:        #00e5c3;
    --teal-dim:    rgba(0,229,195,0.12);
    --red:         #ff4d6d;
    --red-dim:     rgba(255,77,109,0.12);
    --green:       #23d05e;
    --green-dim:   rgba(35,208,94,0.12);
    --amber:       #f5c842;
    --text-main:   #e8edf5;
    --text-muted:  #5a6a82;
    --shadow:      0 24px 64px rgba(0,0,0,0.7);
}

/* ── Page ── */
.stApp {
    background:
        radial-gradient(ellipse at 15% 0%,   rgba(0,229,195,0.06) 0%, transparent 55%),
        radial-gradient(ellipse at 90% 80%,  rgba(255,77,109,0.05) 0%, transparent 55%),
        #050a12;
    font-family: 'Outfit', sans-serif;
    color: var(--text-main);
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Hero banner ── */
.hero {
    position: relative;
    text-align: center;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    border-radius: 24px;
    border: 1px solid var(--border);
    background:
        linear-gradient(135deg,
            rgba(0,229,195,0.10) 0%,
            rgba(0,0,0,0) 50%,
            rgba(255,77,109,0.07) 100%);
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.06);
    overflow: hidden;
    animation: fadeDown .6s ease both;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 38px,
        rgba(255,255,255,0.015) 38px,
        rgba(255,255,255,0.015) 39px
    );
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    letter-spacing: .18em;
    color: var(--teal);
    text-transform: uppercase;
    margin-bottom: .8rem;
    padding: .25rem .9rem;
    border: 1px solid rgba(0,229,195,0.3);
    border-radius: 50px;
    background: rgba(0,229,195,0.07);
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -.02em;
    margin: 0 0 .6rem;
    line-height: 1.15;
}
.hero h1 span {
    background: linear-gradient(90deg, var(--teal), #4fffea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: var(--text-muted);
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
}

/* ── Card ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.6rem;
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    animation: fadeUp .5s ease both;
    transform: perspective(1000px) rotateX(.8deg);
    transition: transform .35s ease, box-shadow .35s ease;
}
.card:hover {
    transform: perspective(1000px) rotateX(0deg) translateY(-3px);
    box-shadow: 0 32px 80px rgba(0,0,0,0.75), 0 0 0 1px var(--border-glow);
}

/* ── Card title ── */
.card-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-main);
    letter-spacing: .04em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: .4rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,229,195,0.25) !important;
    border-radius: 14px !important;
    background: var(--teal-dim) !important;
    transition: border-color .2s ease, background .2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--teal) !important;
    background: rgba(0,229,195,0.16) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-muted) !important;
    font-size: .9rem !important;
}

/* ── Scan line animation on image ── */
.scan-wrapper {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 0 40px rgba(0,229,195,0.15);
}
.scan-line {
    position: absolute;
    top: -4px;
    left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
    animation: scan 2.5s ease-in-out infinite;
    z-index: 10;
    box-shadow: 0 0 12px var(--teal);
}
@keyframes scan {
    0%   { top: 0%; opacity: 1; }
    95%  { top: 100%; opacity: .6; }
    100% { top: 100%; opacity: 0; }
}

/* ── Result boxes ── */
.result-normal {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    background: var(--green-dim);
    border: 1px solid rgba(35,208,94,0.35);
    border-left: 5px solid var(--green);
    padding: 1.4rem 1.6rem;
    border-radius: 16px;
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--green);
    box-shadow: 0 10px 30px rgba(35,208,94,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
    animation: popIn .4s cubic-bezier(.34,1.56,.64,1) both;
    transform: perspective(600px) rotateY(-1deg);
    transition: transform .3s ease;
}
.result-normal:hover { transform: perspective(600px) rotateY(0deg) scale(1.01); }

.result-danger {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    background: var(--red-dim);
    border: 1px solid rgba(255,77,109,0.35);
    border-left: 5px solid var(--red);
    padding: 1.4rem 1.6rem;
    border-radius: 16px;
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--red);
    box-shadow: 0 10px 30px rgba(255,77,109,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
    animation: popIn .4s cubic-bezier(.34,1.56,.64,1) both;
    transform: perspective(600px) rotateY(-1deg);
    transition: transform .3s ease;
}
.result-danger:hover { transform: perspective(600px) rotateY(0deg) scale(1.01); }

.result-icon { font-size: 2.2rem; }
.result-label { font-size: .72rem; font-weight: 500; letter-spacing: .12em; opacity: .7; text-transform: uppercase; }
.result-value { font-size: 1.5rem; font-weight: 800; line-height: 1; }
.result-conf  {
    margin-left: auto;
    text-align: right;
    font-family: 'DM Mono', monospace;
    font-size: 1.9rem;
    font-weight: 500;
    opacity: .85;
}

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: .6rem 0;
}
.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: .82rem;
    color: var(--text-muted);
    width: 110px;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    height: 10px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    overflow: hidden;
}
.prob-bar-fill-green {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #23d05e, #4fffb0);
    box-shadow: 0 0 8px rgba(35,208,94,0.5);
    transition: width .8s cubic-bezier(.4,0,.2,1);
}
.prob-bar-fill-red {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #ff4d6d, #ff8fa3);
    box-shadow: 0 0 8px rgba(255,77,109,0.5);
    transition: width .8s cubic-bezier(.4,0,.2,1);
}
.prob-pct {
    font-family: 'DM Mono', monospace;
    font-size: .82rem;
    color: var(--text-main);
    width: 52px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Disclaimer box ── */
.disclaimer {
    background: rgba(245,200,66,0.07);
    border: 1px solid rgba(245,200,66,0.25);
    border-left: 4px solid var(--amber);
    border-radius: 12px;
    padding: .85rem 1.1rem;
    font-size: .82rem;
    color: #c8a800;
    line-height: 1.6;
    margin-top: 1.2rem;
}

/* ── Animations ── */
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes popIn {
    from { opacity: 0; transform: scale(.92); }
    to   { opacity: 1; transform: scale(1); }
}

/* ── Landscape layout tweaks ── */

/* Compact hero for wide layout */
.hero {
    padding: 1.6rem 2.5rem !important;
    margin-bottom: 1.4rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    text-align: left !important;
    gap: 2rem;
}
.hero-left { flex: 1; }
.hero-right {
    display: flex;
    gap: .8rem;
    flex-shrink: 0;
}
.hero h1 {
    font-size: 1.9rem !important;
    margin-bottom: .25rem !important;
    white-space: nowrap;
}
.hero p { font-size: .88rem !important; }

/* Stat pill in hero */
.stat-pill {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: .7rem 1.2rem;
    min-width: 90px;
}
.stat-pill .sp-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--teal);
}
.stat-pill .sp-lbl {
    font-size: .65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: .09em;
    margin-top: .15rem;
}

/* Left panel — image viewer */
.panel-img {
    position: sticky;
    top: 1.5rem;
}

/* Upload zone compact */
.upload-compact [data-testid="stFileUploader"] > div {
    padding: 1rem !important;
}

/* Right panel scrollable results */
.panel-results { }

/* Confidence ring (big number display) */
.conf-display {
    display: flex;
    align-items: center;
    gap: 1.4rem;
    margin: .8rem 0 1.2rem;
}
.conf-ring {
    width: 90px; height: 90px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'DM Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    flex-shrink: 0;
    position: relative;
}
.conf-ring-normal {
    background: radial-gradient(circle, rgba(35,208,94,0.18), rgba(35,208,94,0.04));
    border: 2px solid rgba(35,208,94,0.5);
    color: var(--green);
    box-shadow: 0 0 30px rgba(35,208,94,0.25), inset 0 0 20px rgba(35,208,94,0.08);
}
.conf-ring-danger {
    background: radial-gradient(circle, rgba(255,77,109,0.18), rgba(255,77,109,0.04));
    border: 2px solid rgba(255,77,109,0.5);
    color: var(--red);
    box-shadow: 0 0 30px rgba(255,77,109,0.25), inset 0 0 20px rgba(255,77,109,0.08);
}
.conf-meta { flex: 1; }
.conf-meta .cm-badge {
    display: inline-block;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    padding: .2rem .8rem;
    border-radius: 50px;
    margin-bottom: .5rem;
}
.badge-normal { background: rgba(35,208,94,0.15); color: var(--green); border: 1px solid rgba(35,208,94,0.3); }
.badge-danger { background: rgba(255,77,109,0.15); color: var(--red);   border: 1px solid rgba(255,77,109,0.3); }
.conf-meta .cm-title { font-size: 1.4rem; font-weight: 800; color: var(--text-main); line-height: 1.1; }
.conf-meta .cm-sub   { font-size: .82rem; color: var(--text-muted); margin-top: .2rem; }


    text-align: center;
    color: var(--text-muted);
    font-size: .75rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: .06em;
    padding: 1rem 0 2rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LOAD MODEL  (cached — loads only once)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.exists():
        st.error(
            f"❌ Model not found at: `{MODEL_PATH}`\n\n"
            "Run your training pipeline first to generate the model file."
        )
        st.stop()

    model = Net().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


model, device = load_model()

# ──────────────────────────────────────────────
# IMAGE PREPROCESSING
# ──────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize → normalise → convert to batched tensor."""
    image      = image.resize((RESIZE, RESIZE))
    img_array  = np.array(image) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)

    mean = torch.tensor(NORMALIZE_LIST_1).view(3, 1, 1)
    std  = torch.tensor(NORMALIZE_LIST_2).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor.unsqueeze(0)   # shape: [1, 3, H, W]


def is_likely_xray(image: Image.Image) -> bool:
    """
    Basic validation guard to reject obvious non-X-ray images.
    Chest X-rays are usually mostly grayscale and have meaningful contrast.
    This is not a medical validator; it only prevents normal photos/screenshots
    from being sent directly to the pneumonia classifier.
    """
    img = image.resize((224, 224)).convert("RGB")
    arr = np.array(img).astype(np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Non-X-ray photos usually have stronger color channel differences.
    color_diff = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))

    # Very flat or blank images should also be rejected.
    gray = np.mean(arr, axis=2)
    contrast = np.std(gray)

    return color_diff < 25 and contrast > 20

# ──────────────────────────────────────────────
# HERO BANNER  — compact horizontal strip
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <div class="hero-eyebrow">AI · Medical Imaging · PyTorch</div>
        <h1>🩻 Chest X-Ray <span>Diagnostic System</span></h1>
        <p>Deep-learning powered pneumonia detection from chest radiograph images</p>
    </div>
    <div class="hero-right">
        <div class="stat-pill">
            <div class="sp-val">2</div>
            <div class="sp-lbl">Classes</div>
        </div>
        <div class="stat-pill">
            <div class="sp-val">CNN</div>
            <div class="sp-lbl">Architecture</div>
        </div>

    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LANDSCAPE LAYOUT  — left: image | right: results
# ──────────────────────────────────────────────
col_img, col_results = st.columns([1, 1], gap="large")

# ════════════════════════════════════════
# LEFT PANEL — Upload + Image Preview
# ════════════════════════════════════════
with col_img:

    # ── Upload card ──
    st.markdown('<div class="card upload-compact">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 Upload X-Ray Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag & drop or click — JPG / PNG",
        type=["jpg", "jpeg", "png"],
        help="Upload a frontal chest radiograph. The model accepts standard JPG or PNG files.",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Image preview (shown only after upload) ──
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown('<div class="card panel-img">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🖼 Radiograph Preview</div>', unsafe_allow_html=True)
        st.markdown('<div class="scan-wrapper"><div class="scan-line"></div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)   # close scan-wrapper
        st.markdown("</div>", unsafe_allow_html=True)  # close card

# ════════════════════════════════════════
# RIGHT PANEL — Inference + Results
# ════════════════════════════════════════
with col_results:

    if not uploaded_file:
        # ── Placeholder when no image uploaded ──
        st.markdown("""
        <div class="card" style="text-align:center; padding: 4rem 2rem; border-style: dashed;">
            <div style="font-size:3.5rem; margin-bottom:1rem; opacity:.3;">🩺</div>
            <div style="color:var(--text-muted); font-size:.95rem; line-height:1.8;">
                Upload a chest X-ray image<br>on the left to begin analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Validate uploaded image before inference ──
        if not is_likely_xray(image):
            st.markdown("""
            <div class="card" style="border-left:5px solid var(--amber);">
                <div class="card-title">⚠️ Invalid Image</div>
                <div style="color:var(--text-main); font-size:1rem; line-height:1.7;">
                    This does not appear to be a chest X-ray image.<br>
                    Please upload a valid frontal chest radiograph.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # ── Run inference ──
        input_tensor = preprocess_image(image).to(device)

        with st.spinner("🔬 Analysing radiograph..."):
            with torch.no_grad():
                output        = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                prediction    = torch.argmax(probabilities, dim=1).item()

        label_map      = {0: "Normal", 1: "Pneumonia"}
        result         = label_map[prediction]
        confidence     = probabilities[0][prediction].item() * 100
        prob_normal    = probabilities[0][0].item() * 100
        prob_pneumonia = probabilities[0][1].item() * 100

        # ── Diagnosis result card ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🧪 Diagnosis Result</div>', unsafe_allow_html=True)

        if result == "Normal":
            st.markdown(f"""
            <div class="conf-display">
                <div class="conf-ring conf-ring-normal">{confidence:.0f}%</div>
                <div class="conf-meta">
                    <span class="cm-badge badge-normal">✅ Prediction</span>
                    <div class="cm-title">Normal</div>
                    <div class="cm-sub">No signs of pneumonia detected in this radiograph.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="conf-display">
                <div class="conf-ring conf-ring-danger">{confidence:.0f}%</div>
                <div class="conf-meta">
                    <span class="cm-badge badge-danger">⚠️ Prediction</span>
                    <div class="cm-title">Pneumonia Detected</div>
                    <div class="cm-sub">Radiograph shows patterns consistent with pneumonia.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Result box (coloured banner) ──
        if result == "Normal":
            st.markdown(f"""
            <div class="result-normal">
                <div class="result-icon">✅</div>
                <div>
                    <div class="result-label">Diagnosis</div>
                    <div class="result-value">Normal</div>
                </div>
                <div class="result-conf">{confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-danger">
                <div class="result-icon">⚠️</div>
                <div>
                    <div class="result-label">Diagnosis</div>
                    <div class="result-value">Pneumonia Detected</div>
                </div>
                <div class="result-conf">{confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close diagnosis card

        # ── Probability breakdown card ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Probability Breakdown</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label">Normal</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill-green" style="width:{prob_normal:.1f}%"></div>
            </div>
            <div class="prob-pct">{prob_normal:.1f}%</div>
        </div>
        <div class="prob-row">
            <div class="prob-label">Pneumonia</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill-red" style="width:{prob_pneumonia:.1f}%"></div>
            </div>
            <div class="prob-pct">{prob_pneumonia:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Medical disclaimer ──
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and research
            purposes only. It is <strong>not a substitute</strong> for professional medical
            diagnosis. Always consult a licensed radiologist or physician for clinical decisions.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close prob card

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer">
    © 2026 &nbsp;·&nbsp; AI Medical Imaging Demo &nbsp;·&nbsp; Built with PyTorch & Streamlit
</div>
""", unsafe_allow_html=True)