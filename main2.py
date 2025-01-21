import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone

# Load predefined areas and their names
try:
    with open("freedomtech", "rb") as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
except FileNotFoundError:
    print("Saved area data not found. Please define areas before running the detection.")
    exit()

# Load COCO class list
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Open video source
cap = cv2.VideoCapture('easy1.mp4')  # Replace with 0 for webcam, or another video file

count = 0

while True:
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

        # Check if the detected object is a car
        if 'car' in class_name:
            # Calculate the car's center point
            car_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            current_car_centers.append(car_center)

            # Check if the car falls within any predefined areas
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

    # Display car count and free space
    cvzone.putTextRect(frame, f'CAR COUNT: {car_count}', (50, 50), scale=2, thickness=2, colorR=(255, 0, 0))
    cvzone.putTextRect(frame, f'FREE SPACE: {free_space}', (50, 100), scale=2, thickness=2, colorR=(0, 255, 0))

    # Show frame
    cv2.imshow('FRAME', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
