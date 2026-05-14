import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import tempfile

# =========================
# LOAD MODEL
# =========================
model = YOLO("runs/detect/train12/weights/best.pt")

# =========================
# DISTANCE FUNCTION
# =========================
def estimate_distance(box_width, img_width):
    relative_size = box_width / img_width

    if relative_size > 0.5:
        return 5
    elif relative_size > 0.3:
        return 10
    elif relative_size > 0.2:
        return 20
    elif relative_size > 0.1:
        return 50
    else:
        return 100

# =========================
# TEXT TO SPEECH FUNCTION
# =========================
def speak(text):
    tts = gTTS(text)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)

    audio_file = open(tmp_file.name, "rb")
    st.audio(audio_file.read(), format="audio/mp3")

# =========================
# UI
# =========================
st.title("🚦 Traffic Sign Detection (YOLOv8)")

option = st.sidebar.selectbox(
    "Choose Mode",
    ["Upload Image", "Live Webcam"]
)

# =========================
# 📂 IMAGE UPLOAD MODE
# =========================
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        # 🔍 Prediction
        results = model(img)

        boxes = results[0].boxes
        img_height, img_width = img.shape[:2]

        detected_info = []

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                x1, y1, x2, y2 = box.xyxy[0]
                box_width = x2 - x1

                distance = estimate_distance(box_width, img_width)

                detected_info.append(f"{label} at {distance} meters")

        detected_info = list(set(detected_info))

        if detected_info:
            sentence = "Detected: " + ", ".join(detected_info)
        else:
            sentence = "No traffic signs detected"

        # 📝 Show text
        st.subheader("Result")
        st.write(sentence)

        # 🔊 Speak
        speak(sentence)

        # 🖼 Show annotated image
        annotated_img = results[0].plot(labels=True, conf=True)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        st.image(annotated_img, caption="Detected Output", use_column_width=True)

# =========================
# 🎥 LIVE WEBCAM MODE
# =========================
elif option == "Live Webcam":
    st.write("Capture image using webcam")

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        np_arr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 🔍 Prediction
        results = model(img)

        boxes = results[0].boxes
        img_height, img_width = img.shape[:2]

        detected_info = []

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                x1, y1, x2, y2 = box.xyxy[0]
                box_width = x2 - x1

                distance = estimate_distance(box_width, img_width)

                detected_info.append(f"{label} at {distance} meters")

        detected_info = list(set(detected_info))

        if detected_info:
            sentence = "Detected: " + ", ".join(detected_info)
        else:
            sentence = "No traffic signs detected"

        # 📝 Show text
        st.subheader("Result")
        st.write(sentence)

        # 🔊 Speak
        speak(sentence)

        # 🖼 Show annotated image
        annotated_img = results[0].plot(labels=True, conf=True)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        st.image(annotated_img, caption="Detected Output", use_column_width=True)