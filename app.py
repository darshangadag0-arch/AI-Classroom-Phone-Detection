import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import textwrap


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
# HTML HELPER
# Prevents Streamlit from showing HTML as code
# =========================================================

def html(content):
    st.markdown(
        textwrap.dedent(content),
        unsafe_allow_html=True
    )


# =========================================================
# CUSTOM CSS
# PPT-INSPIRED DESIGN
# White + Blue/Grey + Orange
# =========================================================

st.markdown("""
<style>

    /* ================= GLOBAL ================= */

    .stApp {
        background: #FAFAF8;
        color: #20252B;
    }

    .main {
        background: #FAFAF8;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* ================= DOT BACKGROUND ================= */

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.16;
        z-index: 0;

        background-image:
            radial-gradient(#7C9AAA 0.7px, transparent 0.7px);

        background-size: 14px 14px;
    }


    .main > div {
        position: relative;
        z-index: 1;
    }


    /* ================= HEADER ================= */

    .hero {
        background: rgba(255, 255, 255, 0.97);
        border: 1.5px solid #8AA6B5;
        border-radius: 18px;
        padding: 30px 34px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;

        box-shadow:
            0 5px 18px rgba(35, 55, 65, 0.06);
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
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.8px;
        color: #687780;
        margin-bottom: 9px;
    }


    .hero-title {
        font-size: 36px;
        line-height: 1.15;
        font-weight: 800;
        color: #20252B;
        margin-bottom: 10px;
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


    /* ================= SECTION TITLE ================= */

    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: #20252B;
        margin-top: 12px;
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


    /* ================= FLOW CARDS ================= */

    .flow-box {
        background: rgba(255, 255, 255, 0.97);
        border: 1px solid #B5C7D0;
        border-radius: 14px;
        padding: 20px 12px;
        text-align: center;
        min-height: 95px;

        box-shadow:
            0 4px 14px rgba(35, 55, 65, 0.05);
    }


    .flow-number {
        color: #D88A20;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1px;
    }


    .flow-title {
        color: #20252B;
        font-size: 14px;
        font-weight: 750;
        margin-top: 8px;
    }


    /* ================= RESULT CARD ================= */

    .result-card {
        background: white;
        border: 1.5px solid #91AAB8;
        border-radius: 16px;
        padding: 23px;

        box-shadow:
            0 5px 18px rgba(35, 55, 65, 0.07);
    }


    .result-label {
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #71818A;
        margin-bottom: 8px;
    }


    .result-number {
        font-size: 42px;
        font-weight: 850;
        color: #20252B;
        line-height: 1;
    }


    .result-description {
        margin-top: 9px;
        color: #68747C;
        font-size: 14px;
        line-height: 1.5;
    }


    /* ================= DETECTED STATUS ================= */

    .status-detected {
        background: #FFF5E8;
        border: 1.5px solid #E9A23B;
        border-radius: 14px;
        padding: 18px;
        margin-top: 16px;
    }


    .status-detected .status-title {
        font-size: 19px;
        font-weight: 800;
        color: #B86D08;
        margin-bottom: 6px;
    }


    .status-normal {
        background: #F3F8FA;
        border: 1.5px solid #91AAB8;
        border-radius: 14px;
        padding: 18px;
        margin-top: 16px;
    }


    .status-normal .status-title {
        font-size: 19px;
        font-weight: 800;
        color: #405963;
        margin-bottom: 6px;
    }


    .status-text {
        font-size: 14px;
        color: #66727A;
        line-height: 1.5;
    }


    /* ================= MODEL BADGE ================= */

    .model-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: #F4F7F8;
        border: 1px solid #AFC0C8;
        color: #44535B;
        font-size: 12px;
        font-weight: 700;
    }


    /* ================= SIDEBAR ================= */

    section[data-testid="stSidebar"] {
        background: #F3F5F5;
        border-right: 1px solid #C2D0D6;
    }


    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #20252B;
    }


    /* ================= TABS ================= */

    button[data-baseweb="tab"] {
        font-weight: 700;
        color: #59666D;
    }


    button[data-baseweb="tab"][aria-selected="true"] {
        color: #D88A20;
    }


    /* ================= FOOTER ================= */

    .footer {
        text-align: center;
        margin-top: 40px;
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

html("""
<div class="hero">

    <div class="hero-label">
        AI VISION LAB · CLASSROOM COMPUTER VISION
    </div>

    <div class="hero-title">
        AI-Based <span>Phone Detection</span> in Classroom
    </div>

    <div class="hero-subtitle">
        Computer vision based mobile phone detection system
        using YOLOv8 object detection.
    </div>

</div>
""")


# =========================================================
# LOAD YOLO MODEL
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

    html("""
    <div class="model-badge">
        YOLOv8 Medium · yolov8m.pt
    </div>
    """)

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
# DETECTION FLOW
# =========================================================

html("""
<div class="section-title">
    Detection Flow
</div>
""")


flow1, flow2, flow3, flow4, flow5 = st.columns(5)


with flow1:

    html("""
    <div class="flow-box">
        <div class="flow-number">01</div>
        <div class="flow-title">Input Image</div>
    </div>
    """)


with flow2:

    html("""
    <div class="flow-box">
        <div class="flow-number">02</div>
        <div class="flow-title">Image Processing</div>
    </div>
    """)


with flow3:

    html("""
    <div class="flow-box">
        <div class="flow-number">03</div>
        <div class="flow-title">YOLOv8</div>
    </div>
    """)


with flow4:

    html("""
    <div class="flow-box">
        <div class="flow-number">04</div>
        <div class="flow-title">Phone Detection</div>
    </div>
    """)


with flow5:

    html("""
    <div class="flow-box">
        <div class="flow-number">05</div>
        <div class="flow-title">Detection Result</div>
    </div>
    """)


st.write("")
st.write("")


# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs([
    "📷 Camera Snapshot",
    "📁 Upload Classroom Image"
])


# =========================================================
# TAB 1 — CAMERA SNAPSHOT
# =========================================================

with tab1:

    html("""
    <div class="section-title">
        Camera Snapshot
    </div>
    """)

    camera_image = st.camera_input(
        "Capture a classroom image"
    )

    if camera_image is not None:

        image = Image.open(camera_image).convert("RGB")

        img_array = np.array(image)

        # YOLO Prediction
        results = model.predict(
            img_array,
            conf=conf_thresh,
            classes=[PHONE_CLASS_ID],
            verbose=False
        )

        boxes = results[0].boxes

        annotated_img = results[0].plot()


        col1, col2 = st.columns(
            [3, 2],
            gap="large"
        )


        # ---------------- IMAGE ----------------

        with col1:

            html("""
            <div class="section-title">
                Detection Analysis
            </div>
            """)

            st.image(
                annotated_img,
                use_container_width=True
            )


        # ---------------- RESULTS ----------------

        with col2:

            html("""
            <div class="section-title">
                Detection Result
            </div>
            """)


            html(f"""
            <div class="result-card">

                <div class="result-label">
                    DEVICES DETECTED
                </div>

                <div class="result-number">
                    {len(boxes)}
                </div>

                <div class="result-description">
                    Visible mobile phones detected
                    in this classroom image.
                </div>

            </div>
            """)


            # ================= PHONE DETECTED =================

            if len(boxes) > 0:

                html(f"""
                <div class="status-detected">

                    <div class="status-title">
                        📱 Phone Detected
                    </div>

                    <div class="status-text">
                        {len(boxes)} visible mobile phone(s)
                        detected in the classroom image.
                    </div>

                </div>
                """)


                st.write("")

                st.markdown("### Confidence")


                for i, box in enumerate(boxes):

                    confidence = float(
                        box.conf[0]
                    ) * 100


                    st.progress(
                        confidence / 100,
                        text=(
                            f"Phone #{i + 1} · "
                            f"{confidence:.1f}%"
                        )
                    )


            # ================= NO PHONE =================

            else:

                html("""
                <div class="status-normal">

                    <div class="status-title">
                        ✓ No Phone Detected
                    </div>

                    <div class="status-text">
                        No visible mobile phone was
                        detected in this classroom image.
                    </div>

                </div>
                """)


# =========================================================
# TAB 2 — UPLOAD IMAGE
# =========================================================

with tab2:

    html("""
    <div class="section-title">
        Upload Classroom Image
    </div>
    """)


    uploaded_file = st.file_uploader(
        "Select a classroom image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        img_array = np.array(image)


        # YOLO Prediction

        results = model.predict(
            img_array,
            conf=conf_thresh,
            classes=[PHONE_CLASS_ID],
            verbose=False
        )


        boxes = results[0].boxes

        annotated_img = results[0].plot()


        col1, col2 = st.columns(
            [3, 2],
            gap="large"
        )


        # ---------------- IMAGE ----------------

        with col1:

            html("""
            <div class="section-title">
                Detection Analysis
            </div>
            """)


            st.image(
                annotated_img,
                use_container_width=True
            )


        # ---------------- RESULTS ----------------

        with col2:

            html("""
            <div class="section-title">
                Detection Result
            </div>
            """)


            html(f"""
            <div class="result-card">

                <div class="result-label">
                    DEVICES DETECTED
                </div>

                <div class="result-number">
                    {len(boxes)}
                </div>

                <div class="result-description">
                    Visible mobile phones detected
                    in this uploaded image.
                </div>

            </div>
            """)


            # ================= PHONE DETECTED =================

            if len(boxes) > 0:

                html("""
                <div class="status-detected">

                    <div class="status-title">
                        📱 Phone Detected
                    </div>

                    <div class="status-text">
                        The model detected visible
                        mobile phone(s) in the
                        uploaded classroom image.
                    </div>

                </div>
                """)


                st.write("")

                st.markdown("### Confidence")


                for i, box in enumerate(boxes):

                    confidence = float(
                        box.conf[0]
                    ) * 100


                    st.progress(
                        confidence / 100,
                        text=(
                            f"Phone #{i + 1} · "
                            f"{confidence:.1f}%"
                        )
                    )


            # ================= NO PHONE =================

            else:

                html("""
                <div class="status-normal">

                    <div class="status-title">
                        ✓ No Phone Detected
                    </div>

                    <div class="status-text">
                        No visible mobile phone was
                        detected in the uploaded image.
                    </div>

                </div>
                """)


# =========================================================
# FOOTER
# =========================================================

html("""
<div class="footer">

    AI Vision Lab · AI-Based Phone Detection in Classroom

    <br>

    Diploma in Artificial Intelligence & Machine Learning

</div>
""")
