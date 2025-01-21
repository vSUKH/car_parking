import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone
import streamlit as st

# Load predefined areas and their names
try:
    with open("freedomtech", "rb") as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
except FileNotFoundError:
    st.error("Saved area data not found. Please define areas before running the detection.")
    st.stop()

# Load COCO class list
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Streamlit UI setup
st.title("Car Detection and Area Monitoring")
st.sidebar.header("Configuration")
video_source = st.sidebar.selectbox("Select Video Source", ["easy1.mp4", "Webcam"], index=0)

# Stop button outside the loop
stop_button = st.button("Stop")  # Define the button once

# Open video source
cap = cv2.VideoCapture(0 if video_source == "Webcam" else video_source)

frame_display = st.image([])  # Placeholder for the video frame
car_count_display = st.metric("Car Count", 0)
free_space_display = st.metric("Free Spaces", len(polylines))

count = 0

while cap.isOpened():
    if stop_button:  # Check if the stop button is clicked
        st.write("Video processing stopped.")
        break

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
        continue

    count += 1
    if count % 3 != 0:  # Skip every 3rd frame for efficiency
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    detections = results[0].boxes.data  # Extract detected boxes
    detections_df = pd.DataFrame(detections).astype("float")

    current_car_centers = []

    for _, row in detections_df.iterrows():
        x1, y1, x2, y2, conf, cls_id = map(int, row[:6])
        class_name = class_list[cls_id]

        if 'car' in class_name:  # If the detected object is a car
            car_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            current_car_centers.append(car_center)

            for i, polygon in enumerate(polylines):
                if cv2.pointPolygonTest(np.array(polygon), car_center, False) >= 0:
                    # Mark cars inside areas
                    cv2.circle(frame, car_center, 5, (255, 0, 0), -1)
                    cvzone.putTextRect(
                        frame,
                        f'Car in {area_names[i]}',
                        (x1, y1 - 10),
                        scale=1,
                        thickness=1,
                        colorR=(0, 255, 0)
                    )
                    break

    # Count cars in each area
    car_count = 0
    for polygon in polylines:
        for center in current_car_centers:
            if cv2.pointPolygonTest(np.array(polygon), center, False) >= 0:
                car_count += 1
                break

    # Calculate free spaces
    free_space = len(polylines) - car_count

    # Draw predefined areas
    for i, polygon in enumerate(polylines):
        cv2.polylines(frame, [np.array(polygon)], True, (0, 0, 255), 2)
        cvzone.putTextRect(
            frame,
            area_names[i],
            tuple(np.array(polygon)[0]),
            scale=1,
            thickness=1,
            colorR=(255, 0, 0)
        )

    # Update metrics and frame
    car_count_display.metric("Car Count", car_count)
    free_space_display.metric("Free Spaces", free_space)

    # Convert the frame to RGB for Streamlit and update the image placeholder
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_display.image(frame)

cap.release()
