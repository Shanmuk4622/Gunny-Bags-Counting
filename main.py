# main.py
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import os
import datetime
import csv
from src.tracker import Tracker  # Updated import

# Load YOLO model
model = YOLO('weights/best.pt')

# Load video
video_path = "data/video1.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file '{video_path}' not found.")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Video file '{video_path}' could not be opened.")

original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(original_fps / 10)

# Load class list
weights_file = "weights/coco1.txt"
if not os.path.exists(weights_file):
    raise FileNotFoundError(f"Class list file '{weights_file}' not found.")

with open(weights_file, "r") as my_file:
    class_list = my_file.read().split("\n")

frame_count = 0
tracker = Tracker()
cy1, cy2 = 200, 300  # Set these according to your video
offset = 10
count = 0
ids = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

    if frame_count % frame_interval != 0:
        frame_count += 1
        continue

    frame_count += 1
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    output_image = frame.copy()

    if not results or not results[0].boxes.data.size:
        print("No objects detected.")
        cv2.imshow("RGB", output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")

    detections = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = row
        detections.append([int(x1), int(y1), int(x2), int(y2)])

    bbox_id = tracker.update(detections)

    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2
        cv2.circle(output_image, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(output_image, str(obj_id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        if cy1 <= cy <= cy2:
            cv2.rectangle(output_image, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if (cy1 - offset < cy < cy1 + offset) or (cy2 - offset < cy < cy2 + offset):
            if obj_id not in ids:
                count += 1
                ids.append(obj_id)
                current_time = datetime.datetime.now().replace(microsecond=0)
                print(f"Count: {count} at: {current_time}")

                # Save the count with date and time to the CSV file
                csv_file_path = "data/count_data.csv"
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([current_time, count])

    # Draw counter lines and count text
    cv2.line(output_image, (100, cy1), (900, cy1), (0, 0, 255), 2)
    cv2.putText(output_image, 'Counter Line 1', (90, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.line(output_image, (100, cy2), (900, cy2), (0, 0, 255), 2)
    cv2.putText(output_image, 'Counter Line 2', (90, cy2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(output_image, f'Count: {count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("RGB", output_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
