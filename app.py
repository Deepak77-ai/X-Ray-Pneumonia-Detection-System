# app.py
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

# -------------------------------------------------
# MODEL ARTIFACT PATH
# -------------------------------------------------
#trained_model_path
#MODEL_PATH = Path("xray/entity/trained_model_path")
MODEL_PATH = Path("xray_model.pth")


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Chest X-Ray Diagnostic System",
    page_icon="🩻",
    layout="centered",
)

# -------------------------------------------------
# GLOBAL CSS
# -------------------------------------------------
st.markdown(
    """
<style>
body {
    background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
}

.main {
    padding-top: 25px;
}

.header {
    text-align: center;
    margin-bottom: 45px;
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(135deg, #1a237e, #6a1b9a);
    box-shadow: 0 25px 60px rgba(0,0,0,0.25);
}

.header h1 {
    font-size: 42px;
    font-weight: 900;
    color: #ffffff;
}

.header p {
    font-size: 18px;
    color: #e1bee7;
}

.card {
    background: rgba(255,255,255,0.95);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 30px;
    box-shadow:
        0 18px 40px rgba(0,0,0,0.12),
        inset 0 1px 0 rgba(255,255,255,0.6);
    animation: fadeUp 0.5s ease;
}

@keyframes fadeUp {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

.card-title {
    font-size: 22px;
    font-weight: 800;
    color: #263238;
    margin-bottom: 18px;
}

.result-normal {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border-left: 6px solid #2e7d32;
    padding: 20px;
    border-radius: 14px;
    font-size: 22px;
    font-weight: 800;
    color: #1b5e20;
    box-shadow: 0 10px 25px rgba(46,125,50,0.25);
}

.result-danger {
    background: linear-gradient(135deg, #ffebee, #ffcdd2);
    border-left: 6px solid #c62828;
    padding: 20px;
    border-radius: 14px;
    font-size: 22px;
    font-weight: 800;
    color: #b71c1c;
    box-shadow: 0 10px 25px rgba(198,40,40,0.25);
}

.footer {
    text-align: center;
    color: #455a64;
    font-size: 14px;
    margin-top: 50px;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# LOAD MODEL FROM ARTIFACT
# -------------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.exists():
        st.error(
            f"Model artifact not found: {MODEL_PATH}\n\n"
            "Please run your training pipeline first so it creates this model file."
        )
        st.stop()

    model = Net().to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model, device


model, device = load_model()

# -------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((RESIZE, RESIZE))

    img_array = np.array(image) / 255.0

    img_tensor = torch.tensor(
        img_array,
        dtype=torch.float32
    ).permute(2, 0, 1)

    mean = torch.tensor(NORMALIZE_LIST_1).view(3, 1, 1)
    std = torch.tensor(NORMALIZE_LIST_2).view(3, 1, 1)

    img_tensor = (img_tensor - mean) / std

    return img_tensor.unsqueeze(0)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
<div class="header">
    <h1>🩻 Chest X-Ray Diagnostic System</h1>
    <p>AI-powered detection of Pneumonia from chest X-ray images</p>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# UPLOAD CARD
# -------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    '<div class="card-title">📤 Upload X-Ray Image</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Select a chest X-ray image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">🖼 Uploaded Image</div>',
        unsafe_allow_html=True,
    )
    st.image(image, width=650)
    st.markdown("</div>", unsafe_allow_html=True)

    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][prediction].item() * 100

    label_map = {
        0: "Normal",
        1: "Pneumonia",
    }

    result = label_map[prediction]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">🧪 Diagnosis Result</div>',
        unsafe_allow_html=True,
    )

    if result == "Normal":
        st.markdown(
            f"""
            <div class="result-normal">
                ✅ NORMAL — Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-danger">
                ⚠️ PNEUMONIA — Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📊 Probability Distribution")

    st.progress(probabilities[0][prediction].item())

    st.write(
        {
            "Normal (%)": f"{probabilities[0][0].item() * 100:.2f}",
            "Pneumonia (%)": f"{probabilities[0][1].item() * 100:.2f}",
        }
    )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    '<div class="footer">© 2026 • AI Medical Imaging Demo • Built with PyTorch & Streamlit</div>',
    unsafe_allow_html=True,
)
