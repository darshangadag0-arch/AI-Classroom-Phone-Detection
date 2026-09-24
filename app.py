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
# PPT-INSPIRED DESIGN
# White + Blue/Grey + Orange
# =========================================================

st.markdown("""
<style>

    /* ---------- MAIN BACKGROUND ---------- */

    .stApp {
        background-color: #FAFAF7;
    }

    .main {
        background-color: #FAFAF7;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- DOT PATTERN ---------- */

    .stApp {
        background-image:
            radial-gradient(
                #AABBC4 0.7px,
                transparent 0.7px
            );
        background-size: 14px 14px;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background-color: #F1F4F5;
        border-right: 1px solid #C4D0D5;
    }


    /* ---------- HEADINGS ---------- */

    h1 {
        color: #20252B !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #20252B !important;
        font-weight: 800 !important;
    }

    h3 {
        color: #30383D !important;
        font-weight: 700 !important;
    }


    /* ---------- TABS ---------- */

    button[data-baseweb="tab"] {
        color: #59666D !important;
        font-weight: 700 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #D88A20 !important;
    }


    /* ---------- METRIC ---------- */

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #AABDC6;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(40, 60, 70, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: #687780 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #20252B !important;
    }


    /* ---------- FILE UPLOADER ---------- */

    section[data-testid="stFileUploaderDropzone"] {
        background-color: white;
        border: 1.5px dashed #9BAFB9;
        border-radius: 14px;
    }


    /* ---------- IMAGE ---------- */

    img {
        border-radius: 12px;
    }


    /* ---------- DIVIDER ---------- */

    hr {
        border-color: #D2DADD;
    }


    /* ---------- BUTTON ---------- */

    .stButton > button {
        border: 1px solid #91AAB8;
        border-radius: 10px;
        background-color: white;
        color: #30383D;
        font-weight: 600;
    }


    /* ---------- SLIDER ---------- */

    div[data-baseweb="slider"] {
        margin-top: 10px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.title("AI-Based Phone Detection in Classroom")

st.caption(
    "AI VISION LAB  •  CLASSROOM COMPUTER VISION"
)

st.write(
    "Computer vision based mobile phone detection system "
    "using YOLOv8 object detection."
)

st.divider()


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

    st.success("Model Loaded")


# =========================================================
# DETECTION FLOW
# =========================================================

st.header("Detection Flow")

flow1, flow2, flow3, flow4, flow5 = st.columns(5)


with flow1:
    st.info("**01**\n\nInput Image")


with flow2:
    st.info("**02**\n\nImage Processing")


with flow3:
    st.info("**03**\n\nYOLOv8")


with flow4:
    st.info("**04**\n\nPhone Detection")


with flow5:
    st.info("**05**\n\nDetection Result")


st.write("")
st.divider()


# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs([
    "📷 Camera Snapshot",
    "📁 Upload Classroom Image"
])


# =========================================================
# TAB 1
# CAMERA SNAPSHOT
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

        # Convert to NumPy
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


        # ---------------------------------------------
        # TWO COLUMN LAYOUT
        # ---------------------------------------------

        col1, col2 = st.columns(
            [3, 2],
            gap="large"
        )


        # ---------------------------------------------
        # LEFT — IMAGE
        # ---------------------------------------------

        with col1:

            st.subheader("Detection Analysis")

            st.image(
                annotated_img,
                use_container_width=True
            )


        # ---------------------------------------------
        # RIGHT — RESULTS
        # ---------------------------------------------

        with col2:

            st.subheader("Detection Result")

            st.metric(
                label="DEVICES DETECTED",
                value=len(boxes)
            )


            # Phone detected
            if len(boxes) > 0:

                st.warning(
                    f"📱 {len(boxes)} visible "
                    f"mobile phone(s) detected."
                )

                st.markdown("### Confidence")

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


            # No phone
            else:

                st.success(
                    "✓ No visible mobile phone "
                    "detected in this image."
                )


# =========================================================
# TAB 2
# UPLOAD IMAGE
# =========================================================

with tab2:

    st.header("Upload Classroom Image")

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

        # Convert to NumPy
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


        # ---------------------------------------------
        # TWO COLUMN LAYOUT
        # ---------------------------------------------

        col1, col2 = st.columns(
            [3, 2],
            gap="large"
        )


        # ---------------------------------------------
        # LEFT — IMAGE
        # ---------------------------------------------

        with col1:

            st.subheader("Detection Analysis")

            st.image(
                annotated_img,
                use_container_width=True
            )


        # ---------------------------------------------
        # RIGHT — RESULTS
        # ---------------------------------------------

        with col2:

            st.subheader("Detection Result")

            st.metric(
                label="DEVICES DETECTED",
                value=len(boxes)
            )


            # Phone detected
            if len(boxes) > 0:

                st.warning(
                    f"📱 {len(boxes)} visible "
                    f"mobile phone(s) detected."
                )

                st.markdown("### Confidence")

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


            # No phone
            else:

                st.success(
                    "✓ No visible mobile phone "
                    "detected in this image."
                )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.header("System Information")

info1, info2, info3 = st.columns(3)


with info1:

    st.subheader("AI Model")

    st.write(
        "YOLOv8 Medium is used for "
        "object detection."
    )


with info2:

    st.subheader("Target Object")

    st.write(
        "The system specifically detects "
        "visible mobile phones."
    )


with info3:

    st.subheader("Output")

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
