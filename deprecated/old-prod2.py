import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import numpy as np
import screeninfo

# Get the primary screen
screen = screeninfo.get_monitors()[0]
width, height = screen.width, screen.height

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('Tracker')
#cv2.setMouseCallback('RGB', RGB)

# Load the YOLO model for object detection (people)
person_model = YOLO("../models/yolo11n.pt")
person_names = person_model.model.names

# Load the YOLO model specifically trained for face detection
face_model = YOLO("../models/yolov8n-face.pt") # You might need to download this model
face_names = face_model.model.names

# Open the video file (use video file or webcam, here using webcam)
rtsp_url = "rtsp://192.168.33.109:554/0/0/0"


cap = cv2.VideoCapture(rtsp_url)
count=0
cy1=261
cy2=286
offset=8
inp={}
enter=[]
exp={}
exitp=[]
while True:
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

#    frame = cv2.resize(frame, (1020, 600))
# Resize the frame to the screen dimensions
    frame = cv2.resize(frame, (width, height))

    # Add the "Demonstration only." text overlay
    text = "This is just a demonstration, we are not recording your biometerics."
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255)  # Red color (BGR)
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = 10
    text_y = 30  # Adjust y-coordinate for vertical position

    cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Run YOLO for person detection
    person_results = person_model.track(frame, persist=True, classes=0)

    # Check if there are any people detected
    if person_results[0].boxes is not None and person_results[0].boxes.id is not None:
        person_boxes = person_results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        person_track_ids = person_results[0].boxes.id.int().cpu().tolist()  # Track IDs

        for person_box, track_id in zip(person_boxes, person_track_ids):
            px1, py1, px2, py2 = person_box
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'Person {track_id}', (px1, py2), 1, 1)

            # Extract the region of interest (ROI) for the detected person
            person_roi = frame[py1:py2, px1:px2]
            if person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                # Run face detection on the person's ROI
                face_results = face_model(person_roi)

                if face_results.xyxy[0] is not None:
                    face_boxes = face_results.xyxy[0].int().cpu().tolist()
                    for fx1, fy1, fx2, fy2 in face_boxes:
                        # Adjust face coordinates to be within the main frame
                        absolute_fx1 = px1 + fx1
                        absolute_fy1 = py1 + fy1
                        absolute_fx2 = px1 + fx2
                        absolute_fy2 = py1 + fy2
                        cv2.rectangle(frame, (absolute_fx1, absolute_fy1), (absolute_fx2, absolute_fy2), (255, 0, 0), 2)
                        cvzone.putTextRect(frame, f'Face', (absolute_fx1, absolute_fy1 - 10), 1, 1, colorR=(255,0,0))


#    cv2.line(frame,(440,286),(1018,286),(0,0,255),2)
#    cv2.line(frame,(438,261),(1018,261),(255,0,255),2)

    cv2.imshow("People Tracker with Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
