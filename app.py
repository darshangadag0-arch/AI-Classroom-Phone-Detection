import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI-Based Phone Detection",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background-color: #FAFAF7 !important;
        color: #20252B !important;

        background-image:
            radial-gradient(
                #AABBC4 0.7px,
                transparent 0.7px
            );

        background-size: 14px 14px;
    }


    .main {
        background-color: #FAFAF7 !important;
    }


    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       ALL GENERAL TEXT
       ===================================================== */

    p {
        color: #303A40 !important;
    }

    label {
        color: #303A40 !important;
    }

    span {
        color: inherit;
    }

    div {
        color: #303A40;
    }


    /* =====================================================
       HEADINGS
       ===================================================== */

    h1 {
        color: #172026 !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #172026 !important;
        font-weight: 800 !important;
    }

    h3 {
        color: #202A30 !important;
        font-weight: 750 !important;
    }

    h4 {
        color: #303A40 !important;
        font-weight: 700 !important;
    }


    /* =====================================================
       CAPTION / SECONDARY TEXT
       ===================================================== */

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #596970 !important;
    }


    /* =====================================================
       MARKDOWN TEXT
       ===================================================== */

    [data-testid="stMarkdownContainer"] p {
        color: #303A40 !important;
    }

    [data-testid="stMarkdownContainer"] li {
        color: #303A40 !important;
    }

    [data-testid="stMarkdownContainer"] strong {
        color: #202A30 !important;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background-color: #F0F4F5 !important;
        border-right: 1px solid #C2D0D6;
    }


    section[data-testid="stSidebar"] * {
        color: #26343B !important;
    }


    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #172026 !important;
    }


    /* =====================================================
       SLIDER
       ===================================================== */

    section[data-testid="stSidebar"] label {
        color: #26343B !important;
        font-weight: 600 !important;
    }


    /* Slider value text */

    div[data-testid="stSlider"] {
        color: #26343B !important;
    }

    div[data-testid="stSlider"] label {
        color: #26343B !important;
    }


    /* =====================================================
       TABS
       ===================================================== */

    button[data-baseweb="tab"] {
        color: #46565E !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }


    button[data-baseweb="tab"][aria-selected="true"] {
        color: #D88A20 !important;
    }


    /* =====================================================
       DIVIDER
       ===================================================== */

    hr {
        border-color: #CCD6DA !important;
    }


    /* =====================================================
       METRIC CARD
       ===================================================== */

    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;

        border: 1px solid #9FB3BD !important;

        border-radius: 14px !important;

        padding: 18px !important;

        box-shadow:
            0 4px 14px rgba(40, 60, 70, 0.07) !important;
    }


    div[data-testid="stMetricLabel"] {
        color: #52636B !important;
        font-weight: 700 !important;
    }


    div[data-testid="stMetricValue"] {
        color: #172026 !important;
        font-weight: 800 !important;
    }


    div[data-testid="stMetricDelta"] {
        color: #52636B !important;
    }


    /* =====================================================
       INFO BOX
       ===================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px !important;
    }


    /* Make alert text readable */

    div[data-testid="stAlert"] p {
        color: #26343B !important;
        font-weight: 600 !important;
    }


    /* =====================================================
       SUCCESS MESSAGE
       ===================================================== */

    div[data-testid="stAlert"][kind="success"] p {
        color: #245A3A !important;
    }


    /* =====================================================
       WARNING MESSAGE
       ===================================================== */

    div[data-testid="stAlert"][kind="warning"] p {
        color: #754A08 !important;
    }


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    section[data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;

        border: 1.5px dashed #91A8B3 !important;

        border-radius: 14px !important;
    }


    section[data-testid="stFileUploaderDropzone"] * {
        color: #34434A !important;
    }


    section[data-testid="stFileUploaderDropzone"] button {
        color: #26343B !important;
        background-color: #F3F6F7 !important;
        border: 1px solid #A5B7BF !important;
    }


    /* =====================================================
       CAMERA INPUT
       ===================================================== */

    div[data-testid="stCameraInput"] {
        color: #303A40 !important;
    }


    div[data-testid="stCameraInput"] * {
        color: #303A40 !important;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        background-color: #FFFFFF !important;

        color: #26343B !important;

        border: 1px solid #91A8B3 !important;

        border-radius: 10px !important;

        font-weight: 650 !important;
    }


    .stButton > button:hover {
        border-color: #D88A20 !important;
        color: #B86D08 !important;
    }


    /* =====================================================
       NUMBER INPUT / TEXT INPUT
       ===================================================== */

    input {
        color: #202A30 !important;
        background-color: #FFFFFF !important;
    }


    textarea {
        color: #202A30 !important;
        background-color: #FFFFFF !important;
    }


    /* =====================================================
       PROGRESS BAR TEXT
       ===================================================== */

    div[data-testid="stProgress"] {
        color: #303A40 !important;
    }


    /* =====================================================
       IMAGE
       ===================================================== */

    img {
        border-radius: 12px;
    }


    /* =====================================================
       EXPANDER
       ===================================================== */

    details {
        background-color: #FFFFFF !important;
        border: 1px solid #C1CED3 !important;
        border-radius: 12px !important;
    }


    details summary {
        color: #26343B !important;
        font-weight: 700 !important;
    }


</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.title(
    "AI-Based Phone Detection in Classroom"
)

st.caption(
    "AI VISION LAB  •  CLASSROOM COMPUTER VISION"
)

st.write(
    "Computer vision based mobile phone detection system "
    "using YOLOv8 object detection."
)

st.divider()


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

    st.header("⚙️ Model Configuration")

    st.write("")

    conf_thresh = st.slider(
        "Detection Confidence",
        min_value=0.15,
        max_value=0.85,
        value=0.25,
        step=0.05
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Model:** YOLOv8 Medium")

    st.write("**File:** `yolov8m.pt`")

    st.write("**Target Class:** Mobile Phone")

    st.write("**Class ID:** 67")

    st.write("**Interface:** Streamlit")

    st.write("**Input:** Image / Camera Snapshot")

    st.divider()

    st.success("✓ Model Loaded")


# =========================================================
# DETECTION FLOW
# =========================================================

st.header("Detection Flow")

flow1, flow2, flow3, flow4, flow5 = st.columns(5)


with flow1:

    st.info(
        "**01**\n\n"
        "Input Image"
    )


with flow2:

    st.info(
        "**02**\n\n"
        "Image Processing"
    )


with flow3:

    st.info(
        "**03**\n\n"
        "YOLOv8"
    )


with flow4:

    st.info(
        "**04**\n\n"
        "Phone Detection"
    )


with flow5:

    st.info(
        "**05**\n\n"
        "Detection Result"
    )


st.write("")

st.divider()


# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs(
    [
        "📷 Camera Snapshot",
        "📁 Upload Classroom Image"
    ]
)


# =========================================================
# CAMERA TAB
# =========================================================

with tab1:

    st.header("Camera Snapshot")

    st.write(
        "Capture a classroom image and analyze it "
        "using the YOLOv8 model."
    )

    camera_image = st.camera_input(
        "Capture Classroom Image"
    )


    if camera_image is not None:

        # Open image
        image = Image.open(
            camera_image
        ).convert("RGB")


        # Convert image
        img_array = np.array(image)


        # YOLO prediction
        results = model.predict(
            img_array,
            conf=conf_thresh,
            classes=[PHONE_CLASS_ID],
            verbose=False
        )


        # Detection boxes
        boxes = results[0].boxes


        # Annotated image
        annotated_img = results[0].plot()


        # -----------------------------------------------
        # COLUMNS
        # -----------------------------------------------

        col1, col2 = st.columns(
            [3, 2],
            gap="large"
        )


        # -----------------------------------------------
        # IMAGE
        # -----------------------------------------------

        with col1:

            st.subheader(
                "Detection Analysis"
            )

            st.image(
                annotated_img,
                use_container_width=True
            )


        # -----------------------------------------------
        # RESULT
        # -----------------------------------------------

        with col2:

            st.subheader(
                "Detection Result"
            )


            st.metric(
                label="DEVICES DETECTED",
                value=len(boxes)
            )


            # PHONE DETECTED

            if len(boxes) > 0:

                st.warning(
                    f"📱 {len(boxes)} visible "
                    f"mobile phone(s) detected."
                )


                st.markdown(
                    "### Confidence"
                )


                for i, box in enumerate(boxes):

                    confidence = (
                        float(box.conf[0]) * 100
                    )


                    st.progress(
                        confidence / 100,
                        text=(
                            f"Phone #{i + 1}  •  "
                            f"{confidence:.1f}%"
                        )
                    )


            # NO PHONE

            else:

                st.success(
                    "✓ No visible mobile phone "
                    "detected in this image."
                )


# =========================================================
# UPLOAD TAB
# =========================================================

with tab2:

    st.header(
        "Upload Classroom Image"
    )

    st.write(
        "Upload a classroom photograph for "
        "mobile phone detection."
    )


    uploaded_file = st.file_uploader(
        "Select an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        # Open image
        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # Convert image
        img_array = np.array(image)


        # YOLO prediction
        results = model.predict(
            img_array,
            conf=conf_thresh,
            classes=[PHONE_CLASS_ID],
            verbose=False
        )


        # Detection boxes
        boxes = results[0].boxes


        # Annotated image
        annotated_img = results[0].plot()


        # -----------------------------------------------
        # COLUMNS
        # -----------------------------------------------

        col1, col2 = st.columns(
            [3, 2],
            gap="large"
        )


        # -----------------------------------------------
        # IMAGE
        # -----------------------------------------------

        with col1:

            st.subheader(
                "Detection Analysis"
            )

            st.image(
                annotated_img,
                use_container_width=True
            )


        # -----------------------------------------------
        # RESULT
        # -----------------------------------------------

        with col2:

            st.subheader(
                "Detection Result"
            )


            st.metric(
                label="DEVICES DETECTED",
                value=len(boxes)
            )


            # PHONE DETECTED

            if len(boxes) > 0:

                st.warning(
                    f"📱 {len(boxes)} visible "
                    f"mobile phone(s) detected."
                )


                st.markdown(
                    "### Confidence"
                )


                for i, box in enumerate(boxes):

                    confidence = (
                        float(box.conf[0]) * 100
                    )


                    st.progress(
                        confidence / 100,
                        text=(
                            f"Phone #{i + 1}  •  "
                            f"{confidence:.1f}%"
                        )
                    )


            # NO PHONE

            else:

                st.success(
                    "✓ No visible mobile phone "
                    "detected in this image."
                )


# =========================================================
# SYSTEM INFORMATION
# =========================================================

st.divider()

st.header(
    "System Information"
)


info1, info2, info3 = st.columns(3)


with info1:

    st.subheader(
        "AI Model"
    )

    st.write(
        "YOLOv8 Medium is used for "
        "object detection."
    )


with info2:

    st.subheader(
        "Target Object"
    )

    st.write(
        "The system specifically detects "
        "visible mobile phones."
    )


with info3:

    st.subheader(
        "Output"
    )

    st.write(
        "The system displays bounding boxes, "
        "device count and confidence."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Based Phone Detection in Classroom  •  "
    "Diploma in Artificial Intelligence & Machine Learning"
)
