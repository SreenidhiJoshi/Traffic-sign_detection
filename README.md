# Traffic Sign Detection and Distance Estimation using YOLOv8

## Overview
This project focuses on real-time traffic sign detection and distance estimation using the YOLOv8 deep learning model. The system detects traffic signs from images or video frames and estimates the approximate distance of the detected sign from the camera using bounding box dimensions.

The project aims to improve road safety and support intelligent transportation systems by providing accurate traffic sign recognition.

---

## Features
- Real-time traffic sign detection
- Distance estimation using bounding box size heuristic
- YOLOv8-based object detection
- Image and video processing support
- High detection accuracy
- Easy-to-use interface using Python

---

## Technologies Used
- Python
- YOLOv8
- OpenCV
- NumPy
- TensorFlow/Keras
- Jupyter Notebook

---

## Project Structure

```bash
Traffic-sign_detection/
│
├── app.py
├── app1.py
├── code (1).ipynb
├── data.yaml
├── results.png
├── yolov8n.pt
├── traffic_sign_model.h5
├── README.md
└── .gitignore
```

---

## Working Principle
1. Input images or video frames are provided to the system.
2. YOLOv8 detects traffic signs present in the frame.
3. Bounding boxes are generated around detected signs.
4. Distance estimation is performed based on bounding box width.
5. Results are displayed with labels and estimated distance.

---

## Distance Estimation
The distance estimation is based on the principle that the apparent size of an object decreases as the distance from the camera increases.

Estimated distance is calculated using:

Distance ∝ 1 / Bounding Box Width

---

## Dataset
The model is trained using a traffic sign dataset containing multiple categories of road signs with labeled annotations.

---

## Applications
- Autonomous vehicles
- Driver assistance systems
- Smart transportation
- Road safety monitoring
- Traffic surveillance systems

---

## Future Enhancements
- Improve distance estimation accuracy
- Deploy on embedded systems
- Real-time video stream optimization
- Add voice alerts for detected signs
- GPS integration for smart navigation

---

## Output
The system detects traffic signs and displays:
- Traffic sign label
- Confidence score
- Estimated distance

---

## Author
Sreenidhi Joshi
