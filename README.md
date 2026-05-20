# AI Vision Intelligence System 🚀

A modern, modular real-time computer vision system designed for object tracking and face recognition. The project utilizes a cascaded neural network architecture leveraging YOLO for object detection/tracking and DeepFace for facial embedding generation.

The system ingests a high-performance MJPEG video stream from a mobile device (via IP Webcam) and serves an interactive web dashboard displaying live analytical metrics.

## ✨ Key Features

- **Two-Stage Cascaded Inference**: Real-time person detection and facial localization utilizing optimized YOLOv8 models.
- **Intelligent Tracking**: Built-in Ultralytics tracking mechanisms to maintain object persistence (`persist=True`) across consecutive frames.
- **Face Recognition**: Features facial embedding extraction via the `Facenet512` architecture. It cross-references a local database using a "lazy evaluation" filter (Lazy Inference triggered every 25 frames) to maximize CPU/GPU efficiency.
- **Asynchronous Web Server**: A lightweight FastAPI backend featuring a dedicated video capture thread (`threading.Thread`) and dynamic page rendering via Jinja2 templates.
- **Real-Time Metrics**: Asynchronous frontend UI updates using the native JavaScript Fetch API to reflect live target data without page reloads.

---

## 🛠️ Tech Stack

- **Backend / Core**: Python 3.12+, FastAPI, Uvicorn, Starlette
- **Computer Vision / ML**: OpenCV, Ultralytics YOLOv8, DeepFace (`Facenet512` backbone, `opencv` detector backend)
- **Frontend / UI**: HTML5, CSS3 (Modern Dark Theme), JavaScript (Fetch API / DOM Manipulation), Jinja2 Templates
- **Package Management**: `uv` (Fast, modern Python package installer and resolver)

---