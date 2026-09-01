import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="GuardianAI | Classroom Vision",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI Design
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #334155;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 15px;
    }
    .violation-card {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 16px;
        margin-top: 10px;
    }
    .safe-card {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid #22c55e;
        border-radius: 10px;
        padding: 16px;
        margin-top: 10px;
    }
    .stat-badge {
        display: inline-block;
        background: #1e293b;
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid #0284c7;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# App Header Banner
st.markdown("""
<div class="header-box">
    <div class="header-title">🛡️ GuardianAI — Classroom Attention Monitor</div>
    <div class="header-subtitle">Real-time deep learning surveillance system for unauthorized mobile device detection during lectures and exams.</div>
</div>
""", unsafe_allow_html=True)

# Load YOLOv8 Medium Model
@st.cache_resource
def load_model():
    return YOLO('yolov8m.pt')

model = load_model()
PHONE_CLASS_ID = 67

# Sidebar Configuration
with st.sidebar:
    st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png", use_container_width=True)
    st.subheader("⚙️ Model Parameters")
    conf_thresh = st.slider("Detection Confidence", min_value=0.15, max_value=0.85, value=0.25, step=0.05)
    st.divider()
    st.markdown("**Active Model:** YOLOv8 Medium (`yolov8m.pt`)")
    st.markdown("**Target Class:** Mobile Device (`Class ID: 67`)")
    st.markdown("**Status:** Online 🟢")

# Tabs Layout
tab1, tab2 = st.tabs(["📸 Live Camera Scan", "📁 Batch Image Analysis"])

# --- Tab 1: Live Snapshot ---
with tab1:
    st.markdown("### 📷 Real-Time Video Snapshot")
    camera_image = st.camera_input("Capture classroom feed")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        img_array = np.array(image)

        results = model.predict(img_array, conf=conf_thresh, classes=[PHONE_CLASS_ID], verbose=False)
        boxes = results[0].boxes
        annotated_img = results[0].plot()

        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(annotated_img, caption="Model Inference Overlay", use_container_width=True)
        with col2:
            st.markdown("### 📊 Diagnostic Results")
            if len(boxes) > 0:
                st.markdown(f"""
                <div class="violation-card">
                    <span class="stat-badge">VIOLATION DETECTED</span>
                    <h3 style="color:#ef4444; margin:0 0 8px 0;">🚨 {len(boxes)} Device(s) in Frame</h3>
                    <p style="color:#fca5a5; font-size:14px; margin:0;">An unauthorized mobile phone usage event was logged by the vision pipeline.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0]) * 100
                    st.progress(conf / 100, text=f"Phone #{i+1} Confidence: {conf:.1f}%")
            else:
                st.markdown("""
                <div class="safe-card">
                    <span class="stat-badge" style="color:#4ade80; border-color:#16a34a;">STATUS NORMAL</span>
                    <h3 style="color:#22c55e; margin:0 0 8px 0;">✅ Classroom Attentive</h3>
                    <p style="color:#86efac; font-size:14px; margin:0;">No unauthorized mobile phone devices detected within the camera boundary.</p>
                </div>
                """, unsafe_allow_html=True)

# --- Tab 2: Upload File ---
with tab2:
    st.markdown("### 📂 Upload Classroom Photo")
    uploaded_file = st.file_uploader("Select an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        results = model.predict(img_array, conf=conf_thresh, classes=[PHONE_CLASS_ID], verbose=False)
        boxes = results[0].boxes
        annotated_img = results[0].plot()

        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(annotated_img, caption="Detection Analysis", use_container_width=True)
        with col2:
            st.markdown("### 📊 Detection Breakdown")
            if len(boxes) > 0:
                st.metric(label="Total Violations Found", value=len(boxes), delta="Action Required", delta_color="inverse")
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0]) * 100
                    st.info(f"Target #{i+1}: Accuracy match at **{conf:.1f}%**")
            else:
                st.metric(label="Total Violations", value=0, delta="Clean")
                st.success("No active device found in this frame.")