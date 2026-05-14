import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train12/weights/best.pt")

st.title("🚦 Traffic Sign Detection using YOLOv8")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prediction
    results = model(img)

    # Annotated image
    annotated_img = results[0].plot(labels=True, conf=True)
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    st.image(annotated_img, caption="Detected Output", use_column_width=True)