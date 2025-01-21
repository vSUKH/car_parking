import cv2
import numpy as np
import cvzone
import pickle

# Load video
cap = cv2.VideoCapture('easy1.mp4')

drawing = False
polylines = []  
area_names = [] 
points = []  
current_name = " "  

# Load saved data, if available
try:
    with open("freedomtech", "rb") as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
except FileNotFoundError:
    print("No saved data found. Starting fresh.")
except Exception as e:
    print(f"Error loading saved data: {e}")

# Mouse callback function for drawing
def draw(event, x, y, flags, param):
    global points, drawing, current_name

    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing a polygon
        points = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:  # Add points to the polygon
        points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:  # Finalize the polygon
        drawing = False
        points.append((x, y))  
        polylines.append(np.array(points, np.int32))  # Save polygon
        current_name = input('Name of area: ')
        area_names.append(current_name if current_name else "Unnamed Area")
        print("Area defined successfully!")
        
        # Prompt to save after defining an area
        save_prompt = input("Do you want to save the defined areas? (y/n): ")
        if save_prompt.lower() == 'y':
            save_data()

# Function to save data
def save_data():
    with open("freedomtech", "wb") as f:
        data = {'polylines': polylines, 'area_names': area_names}
        pickle.dump(data, f)
        print("Data saved successfully.")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Draw saved polygons and names
    for i, polygon in enumerate(polylines):
        cv2.polylines(frame, [polygon], True, (0, 0, 255), 2)
        cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polygon[0]), scale=1, thickness=1)

    # Show frame
    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME', draw)

    # Handle key inputs
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Save the data
        save_data()
    elif key == ord('q'):  # Quit the program
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
