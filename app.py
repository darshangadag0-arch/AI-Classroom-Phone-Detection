import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Vision Lab | Phone Detection",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# PPT-INSPIRED: WHITE + BLUE + ORANGE
# =========================================================

st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #FAFAF8;
        color: #20252B;
    }

    .main {
        background: #FAFAF8;
    }

    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* ---------- TECHNICAL DOT BACKGROUND ---------- */

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.18;
        z-index: 0;

        background-image:
            radial-gradient(#7C9AAA 0.7px, transparent 0.7px);

        background-size: 14px 14px;
    }

    /* Keep content above background */
    .main > div {
        position: relative;
        z-index: 1;
    }

    /* ---------- HEADER ---------- */

    .hero {
        background: rgba(255,255,255,0.96);
        border: 1.5px solid #8AA6B5;
        border-radius: 18px;
        padding: 30px 34px;
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
    }

    .hero::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: #E9A23B;
    }

    .hero-label {
        display: inline-block;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.8px;
        color: #687780;
        margin-bottom: 8px;
    }

    .hero-title {
        font-size: 36px;
        line-height: 1.1;
        font-weight: 800;
        color: #20252B;
        margin-bottom: 8px;
    }

    .hero-title span {
        color: #D88A20;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #66727A;
        line-height: 1.6;
        max-width: 850px;
    }

    /* ---------- SECTION TITLE ---------- */

    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: #20252B;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    .section-title::after {
        content: "";
        display: block;
        width: 55px;
        height: 4px;
        background: #E9A23B;
        border-radius: 4px;
        margin-top: 8px;
    }

    /* ---------- INFO CARDS ---------- */

    .info-card {
        background: rgba(255,255,255,0.96);
        border: 1px solid #B5C7D0;
        border-radius: 14px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 4px 15px rgba(35,55,65,0.06);
    }

    .card-number {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: #E9A23B;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .card-title {
        font-size: 17px;
        font-weight: 750;
        color: #20252B;
        margin-bottom: 6px;
    }

    .card-text {
        font-size: 14px;
        color: #68747C;
        line-height: 1.55;
    }

    /* ---------- DETECTION RESULT ---------- */

    .result-card {
        background: white;
        border: 1.5px solid #91AAB8;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 5px 18px rgba(35,55,65,0.07);
    }

    .result-label {
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #71818A;
        margin-bottom: 7px;
    }

    .result-number {
        font-size: 38px;
        font-weight: 850;
        color: #20252B;
        line-height: 1;
    }

    .result-description {
        margin-top: 8px;
        color: #68747C;
        font-size: 14px;
    }

    /* ---------- DETECTION STATUS ---------- */

    .status-detected {
        background: #FFF5E8;
        border: 1.5px solid #E9A23B;
        border-radius: 14px;
        padding: 18px;
        margin-top: 15px;
    }

    .status-normal {
        background: #F3F8FA;
        border: 1.5px solid #91AAB8;
        border-radius: 14px;
        padding: 18px;
        margin-top: 15px;
    }

    .status-title {
        font-size: 19px;
        font-weight: 800;
        color: #20252B;
        margin-bottom: 5px;
    }

    .status-text {
        font-size: 14px;
        color: #66727A;
        line-height: 1.5;
    }

    /* ---------- MODEL BADGE ---------- */

    .model-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: #F4F7F8;
        border: 1px solid #AFC0C8;
        color: #44535B;
        font-size: 12px;
        font-weight: 700;
        margin-top: 8px;
    }

    /* ---------- FLOW ---------- */

    .flow-box {
        background: white;
        border: 1px solid #AFC0C8;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        height: 100%;
    }

    .flow-number {
        color: #D88A20;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .flow-title {
        color: #20252B;
        font-size: 15px;
        font-weight: 750;
        margin-top: 7px;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #F3F5F5;
        border-right: 1px solid #C2D0D6;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #20252B;
    }

    /* ---------- TABS ---------- */

    button[data-baseweb="tab"] {
        font-weight: 700;
        color: #59666D;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #D88A20;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        border: 1px solid #9BAEB7;
        border-radius: 10px;
        background: white;
        color: #20252B;
        font-weight: 700;
    }

    .stButton > button:hover {
        border-color: #E9A23B;
        color: #D88A20;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        margin-top: 35px;
        padding-top: 18px;
        border-top: 1px solid #D2DADD;
        color: #7A858B;
        font-size: 12px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-label">
        AI VISION LAB · CLASSROOM COMPUTER VISION
    </div>

    <div class="hero-title">
        AI-Based <span>Phone Detection</span> in Classroom
    </div>

    <div class="hero-subtitle">
        Computer vision based mobile phone detection system using
        YOLOv8 object detection.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")


model = load_model()

PHONE_CLASS_ID = 67


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ Model Configuration")

    st.markdown(
        '<div class="model-badge">YOLOv8 Medium · yolov8m.pt</div>',
        unsafe_allow_html=True
    )

    st.write("")

    conf_thresh = st.slider(
        "Detection Confidence",
        min_value=0.15,
        max_value=0.85,
        value=0.25,
        step=0.05
    )

    st.divider()

    st.markdown("### Model Information")

    st.markdown("""
    **Model:** YOLOv8 Medium  
    **Target:** Mobile Phone  
    **Class ID:** 67  
    **Interface:** Streamlit  
    **Input:** Image / Camera Snapshot
    """)

    st.divider()

    st.success("Model Loaded")


# =========================================================
# PROJECT FLOW
# =========================================================

st.markdown(
    '<div class="section-title">Detection Flow</div>',
    unsafe_allow_html=True
)

flow1, flow2, flow3, flow4, flow5 = st.columns(5)

with flow1:
    st.markdown("""
    <div class="flow-box">
        <div class="flow-number">01</div>
        <div class="flow-title">Input Image</div>
    </div>
    """, unsafe_allow_html=True)

with flow2:
    st.markdown("""
    <div class="flow-box">
        <div class="flow-number">02</div>
        <div class="flow-title">Image Processing</div>
    </div>
    """, unsafe_allow_html=True)

with flow3:
    st.markdown("""
    <div class="flow-box">
        <div class="flow-number">03</div>
        <div class="flow-title">YOLOv8</div>
    </div>
    """, unsafe_allow_html=True)

with flow4:
    st.markdown("""
    <div class="flow-box">
        <div class="flow-number">04</div>
        <div class="flow-title">Phone Detection</div>
    </div>
    """, unsafe_allow_html=True)

with flow5:
    st.markdown("""
    <div class="flow-box">
        <div class="flow-number">05</div>
        <div class="flow-title">Detection Result</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs([
    "📷 Camera Snapshot",
    "📁 Upload Classroom Image"
])


# =========================================================
# CAMERA SNAPSHOT
# =========================================================

with tab1:

    st.markdown(
        '<div class="section-title">Camera Snapshot</div>',
        unsafe_allow_html=True
    )

    camera_image = st.camera_input(
        "Capture a classroom image"
    )

    if camera_image is not None:

        image = Image.open(camera_image).convert("RGB")
        img_array = np.array(image)

        results = model.predict(
            img_array,
            conf=conf_thresh,
            classes=[PHONE_CLASS_ID],
            verbose=False
        )

        boxes = results[0].boxes
        annotated_img = results[0].plot()

        col1, col2 = st.columns([3, 2], gap="large")

        # IMAGE
        with col1:

            st.markdown(
                '<div class="section-title">Detection Analysis</div>',
                unsafe_allow_html=True
            )

            st.image(
                annotated_img,
                use_container_width=True
            )

        # RESULT
        with col2:

            st.markdown(
                '<div class="section-title">Detection Result</div>',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            <div class="result-card">

                <div class="result-label">
                    DEVICES DETECTED
                </div>

                <div class="result-number">
                    {len(boxes)}
                </div>

                <div class="result-description">
                    Visible mobile phones detected in this image.
                </div>

            </div>
            """, unsafe_allow_html=True)

            if len(boxes) > 0:

                st.markdown(f"""
                <div class="status-detected">

                    <div class="status-title">
                        📱 Phone Detected
                    </div>

                    <div class="status-text">
                        {len(boxes)} visible mobile phone(s)
                        detected in the classroom image.
                    </div>

                </div>
                """, unsafe_allow_html=True)

                st.write("")

                st.markdown("### Confidence")

                for i, box in enumerate(boxes):

                    conf = float(box.conf[0]) * 100

                    st.progress(
                        conf / 100,
                        text=f"Phone #{i+1} · {conf:.1f}%"
                    )

            else:

                st.markdown("""
                <div class="status-normal">

                    <div class="status-title">
                        ✓ No Phone Detected
                    </div>

                    <div class="status-text">
                        No visible mobile phone was detected
                        in this classroom image.
                    </div>

                </div>
                """, unsafe_allow_html=True)


# =========================================================
# UPLOAD IMAGE
# =========================================================

with tab2:

    st.markdown(
        '<div class="section-title">Upload Classroom Image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Select a classroom image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        results = model.predict(
            img_array,
            conf=conf_thresh,
            classes=[PHONE_CLASS_ID],
            verbose=False
        )

        boxes = results[0].boxes
        annotated_img = results[0].plot()

        col1, col2 = st.columns([3, 2], gap="large")

        with col1:

            st.markdown(
                '<div class="section-title">Detection Analysis</div>',
                unsafe_allow_html=True
            )

            st.image(
                annotated_img,
                use_container_width=True
            )

        with col2:

            st.markdown(
                '<div class="section-title">Detection Result</div>',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            <div class="result-card">

                <div class="result-label">
                    DEVICES DETECTED
                </div>

                <div class="result-number">
                    {len(boxes)}
                </div>

                <div class="result-description">
                    Visible mobile phones detected in this image.
                </div>

            </div>
            """, unsafe_allow_html=True)

            if len(boxes) > 0:

                st.markdown("""
                <div class="status-detected">

                    <div class="status-title">
                        📱 Phone Detected
                    </div>

                    <div class="status-text">
                        The model detected visible mobile phone(s)
                        in the uploaded classroom image.
                    </div>

                </div>
                """, unsafe_allow_html=True)

                st.write("")

                st.markdown("### Confidence")

                for i, box in enumerate(boxes):

                    conf = float(box.conf[0]) * 100

                    st.progress(
                        conf / 100,
                        text=f"Phone #{i+1} · {conf:.1f}%"
                    )

            else:

                st.markdown("""
                <div class="status-normal">

                    <div class="status-title">
                        ✓ No Phone Detected
                    </div>

                    <div class="status-text">
                        No visible mobile phone was detected
                        in the uploaded image.
                    </div>

                </div>
                """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    AI Vision Lab · AI-Based Phone Detection in Classroom
    <br>
    Diploma in Artificial Intelligence & Machine Learning
</div>
""", unsafe_allow_html=True)
