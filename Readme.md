# Gunny Bags Object Detection and Counting System

This project is an object detection and counting system designed to track and count objects (e.g., gunny bags) in a video using the YOLO (You Only Look Once) model. The system processes video frames, detects objects, tracks them, and logs the count along with timestamps in a CSV file.

---

## Demo Video
Watch the demo video on YouTube to see the system in action:

[![ Demo](https://img.youtube.com/vi/f9mZR_FHYyQ/0.jpg)](https://youtu.be/f9mZR_FHYyQ)

---

## Features

- **Object Detection**: Uses the YOLO model to detect objects in video frames.
- **Object Tracking**: Tracks detected objects across frames using a custom tracker.
- **Real-Time Counting**: Counts objects crossing predefined lines in the video.
- **CSV Logging**: Saves the count along with the date and time to a CSV file.
- **Visualization**: Displays the video with bounding boxes, tracking IDs, and counter lines.
- **Customizable**: Easily configurable for different videos and object classes.

---

## Project Structure

```bash
│
├── main.py
├── data/
│   ├── count_data.csv
│   ├── video1.mp4
│   └── video3.mp4
├── weights/
│   ├── best.pt
│   └── coco1.txt
├── src/
│   ├── tracker.py
│   └── __init__.py
├── requirements.txt
└── Readme.md
```


---

## Requirements

To run this project, you need the following dependencies:

- Python 3.8 or higher
- OpenCV (`opencv-python==4.5.5.64`)
- Pandas (`pandas==1.3.5`)
- Ultralytics YOLO (`ultralytics==8.0.20`)
- CVZone (`cvzone==1.5.6`)
- PyTorch (`torch==1.13.1`, `torchvision==0.14.1`)
- NumPy (`numpy==1.21.6`)

Install all dependencies using the following command:

```bash
pip install -r requirements.txt
```

---

## How to Run
1. **Clone the Repository**: Clone this repository to your local machine:
```bash
git clone https://github.com/your-repo-name.git
cd gunny_bags_Count
```

2. **Prepare the Environment**: Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. **Add Your Video:** Place your video file in the `data/` folder and update the video_path variable in `main.py` with the correct file name.

4. **Run the Program:** Execute the `main.py` file:
```bash
python main.py
```
5. **View Results:**

- The video will be displayed with bounding boxes, tracking IDs, and counter lines.
- The count of objects crossing the lines will be logged in `data/count_data.csv`.

---

Configuration
1. Video Input:
Update the video_path variable in main.py to point to your video file:

2. Counter Lines:
Adjust the cy1 and cy2 variables in main.py to set the positions of the counter lines:

3. YOLO Model:
Replace the weights/best.pt file with your trained YOLO model weights if needed.

---

### Output
1. **Video Display:**
- Bounding boxes around detected objects.
- Tracking IDs displayed near objects.
- Counter lines drawn on the video.
2. **CSV Logging:** The `data/count_data.csv` file will contain the count of objects along with timestamps:

```bash
DateTime,Count
2025-04-26 09:50:46,1
2025-04-26 09:50:51,2
2025-04-26 09:50:55,3
```

---

### Troubleshooting
**Video Not Found:** Ensure the video file exists in the `data/` folder and the `video_path` variable is correct.
**Missing Dependencies:** Run `pip install -r requirements.txt` to install all required libraries.
**Incorrect Counter Lines:** Adjust the `cy1` and `cy2` variables to match the video content.

---

### Future Enhancements
- Add support for multiple object classes.
- Implement a web-based dashboard for real-time monitoring.
- Optimize the tracker for better performance with high frame rates.

---

## Acknowledgments
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5) for the object detection model.
- [OpenCV](https://opencv.org/) for video processing.
- [CVZone](https://github.com/cvzone/cvzone) for additional computer vision utilities.