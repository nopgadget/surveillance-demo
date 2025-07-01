# Real-time People Tracker Conference 'Surveillance' Demo

This project is a real-time person tracking demonstration built for conference and educational settings. It uses a YOLO (You Only Look Once) model to identify and track individuals in a live video stream, showcasing the capabilities of modern computer vision in a transparent and responsible manner.

The primary goal of this interactive demo is to engage attendees and raise awareness about data privacy in an era of increasingly powerful surveillance technologies.

---

## How It Works

The application captures a live RTSP video stream and processes it frame by frame to:

1.  **Detect and Track Individuals**: It runs a YOLO object detection model to identify people in the video feed.
2.  **Assign Anonymous IDs**: Each detected person is assigned a temporary, anonymous tracking ID that persists as long as they are in the frame.
3.  **Display Live Feed**: The processed video, complete with bounding boxes and anonymous IDs, is displayed in a full-screen window.
4.  **Overlay Informational Graphics**: A logo and QR code are displayed on the video. Most importantly, a prominent, animated message at the top of the screen continuously informs participants that the demonstration is for illustrative purposes only.

---

## Key Features

* **Real-Time Performance**: Utilizes the `ultralytics` YOLO library for efficient, real-time object detection and tracking.
* **Privacy-First Design**: The system is explicitly designed **not to save any personal data**. A clear on-screen disclaimer runs at all times to ensure informed consent from anyone in view.
* **Interactive Controls**: Allows for on-the-fly interaction to demonstrate the system's capabilities.
* **Informational Overlays**: Displays a company logo, a QR code for more information, and dynamic text to inform viewers about the demo's purpose.

---

## Privacy and Consent

This demonstration is fundamentally a tool for education and awareness. It operates on a principle of full transparency and consent.

**No data is retained, stored, or shared.**

All tracking is performed in real-time, and the session data is discarded the moment the application is closed. The prominent on-screen notifications are a core feature, ensuring that anyone within the camera's view is aware of the technology, its function, and its limitations in this context.

---

## Controls

The application can be controlled using the following keyboard commands while the video window is active:

* **Quit**: Press `q` to close the application and end the video stream.
* **Start/Stop Recording**: Press the `spacebar` to start recording the on-screen display to a local `.mp4` file. Press it again to stop.
* **Save Screenshot**: A brief tap of the `spacebar` will save a single screenshot as a `.jpg` file.

---

## Getting Started

Using a python virtual environment manager of your choice (in this example we will use conda), install the pre-reqs from requirements.txt

```
conda create -n conference_demo python=3.9.6

conda activate conference_demo

# Navigate to the repo

python -m pip install -r .\requirements.txt
```