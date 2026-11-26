
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import time

# =========================
# Configuration
# =========================
MODEL_PATH = Path("models/best.pt")

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide"
)

st.title("NeuroVision AI")
st.subheader("Brain Tumor Detection — YOLOv8")

st.write(
    "Upload an MRI image to detect and localize "
    "potential brain tumor regions."
)

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI")
        st.image(image, use_container_width=True)

    if st.button("Run Detection", type="primary"):

        with st.spinner("Analyzing image..."):

            start_time = time.perf_counter()

            results = model.predict(
                source=image,
                conf=0.25,
                verbose=False
            )

            inference_time = time.perf_counter() - start_time

            result = results[0]

            annotated_image = result.plot()

        with col2:
            st.subheader("Detection Results")
            st.image(
                annotated_image,
                use_container_width=True
            )

        st.metric(
            "Inference Time",
            f"{inference_time:.3f} seconds"
        )

        if len(result.boxes) > 0:

            st.success(
                f"{len(result.boxes)} detection(s) found."
            )

            detections = []

            for box in result.boxes:

                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                class_name = model.names[class_id]

                detections.append({
                    "Class": class_name,
                    "Confidence": f"{confidence:.2%}"
                })

            st.subheader("Detected Classes")

            st.dataframe(
                detections,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning("No detections found.")

st.divider()

st.caption(
    "NeuroVision AI is an educational computer vision project. "
    "Its outputs are not a medical diagnosis."
)
