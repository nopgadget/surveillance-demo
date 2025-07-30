import cv2
import numpy as np
import time
from ultralytics import YOLO
import cvzone
import screeninfo

# Configuration
RTSP_URL = "rtsp://192.168.1.109:554/0/0/0"
MODEL_PATH = "../models/yolo11n.pt"
LOGO_PATH = "../img/odplogo.png"
WINDOW_NAME = "People Tracker"

# Get screen dimensions
screen = screeninfo.get_monitors()[0]
width, height = screen.width, screen.height

# Load logo
logo = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
if logo is None:
    raise FileNotFoundError("Logo image not found!")

# Load QR code
qr_code = cv2.imread("../img/qr-code.png", cv2.IMREAD_UNCHANGED)
if qr_code is None:
    raise FileNotFoundError("QR code image not found!")
    raise FileNotFoundError("Logo image not found!")

# Keep alpha channel if present
# No slicing; let overlay function handle alpha

# Don't resize logo up front; will resize once during overlay based on display size
# logo is kept as-is to preserve quality

# Load YOLO model
model = YOLO(MODEL_PATH)
names = model.model.names

# Open video stream
#cap = cv2.VideoCapture(RTSP_URL)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open stream: {RTSP_URL}")

# Overlay function for non-transparent logos
def overlay_logo_clipped(frame, logo, position="bottom-right", margin=10):
    fh, fw = frame.shape[:2]
    # Use original logo size unless too large for the frame
    max_logo_width = frame.shape[1] // 5
    max_logo_height = frame.shape[0] // 5
    lh, lw = logo.shape[:2]
    if lw > max_logo_width or lh > max_logo_height:
        scale = min(max_logo_width / lw, max_logo_height / lh)
        logo_resized = cv2.resize(logo, (int(lw * scale), int(lh * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4)
    else:
        logo_resized = logo.copy()
    lh, lw = logo_resized.shape[:2]
    lh, lw = logo_resized.shape[:2]

    if position == "top-left":
        x1, y1 = margin, margin
    elif position == "top-right":
        x1, y1 = fw - lw - margin, margin
    elif position == "bottom-left":
        x1, y1 = margin, fh - lh - margin
    else:
        x1, y1 = fw - lw - margin, fh - lh - margin

    x1_clip, y1_clip = max(x1, 0), max(y1, 0)
    x2_clip = min(x1 + lw, fw)
    y2_clip = min(y1 + lh, fh)

    logo_x1 = x1_clip - x1
    logo_y1 = y1_clip - y1
    logo_x2 = logo_x1 + (x2_clip - x1_clip)
    logo_y2 = logo_y1 + (y2_clip - y1_clip)

    cropped_logo = logo_resized[logo_y1:logo_y2, logo_x1:logo_x2]
    roi = frame[y1_clip:y2_clip, x1_clip:x2_clip]

    if cropped_logo.shape[2] == 4:
        logo_rgb = cropped_logo[:, :, :3]
        alpha = cropped_logo[:, :, 3][:, :, np.newaxis] / 255.0

        blended = np.where(alpha > 0, logo_rgb, roi).astype(np.uint8)
        roi[:] = blended
    else:
        roi[:, :] = cropped_logo

    frame[y1_clip:y2_clip, x1_clip:x2_clip] = roi
    return frame

# Main loop
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    results = model.track(frame.copy(), persist=True, classes=0)
    display_frame = frame.copy()

    if results and results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            c = names[class_id]
            x1, y1, x2, y2 = box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(display_frame, f"{track_id}", (x1, y2), 1, 1)
            cvzone.putTextRect(display_frame, f"{c}", (x1, y1), 1, 1)

    # Display informational text
    info_text = "Attention: This demo uses live video only. No data is retained, stored or shared."
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 1  # Reduced font scale for better fit
    font_thickness = 2  # Slightly reduced thickness for slightly less bold text
    text_size, _ = cv2.getTextSize(info_text, font, text_scale, font_thickness)
    text_x = (display_frame.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 20
    fade = 0.5 * (1 + np.sin(time.time() * 2))  # value from 0 to 1
    r = int((1 - fade) * 0 + fade * 173)
    g = int((1 - fade) * 0 + fade * 216)
    b = int((1 - fade) * 0 + fade * 230)
    cv2.putText(display_frame, info_text, (text_x, text_y), font, text_scale, (b, g, r), font_thickness, cv2.LINE_AA)
    # Simulate underline
    underline_y = text_y + 5
    if int(time.time() * 2) % 2 == 0:
        cv2.line(display_frame, (text_x, underline_y), (text_x + text_size[0], underline_y), (0, 0, 0), 2)

    # Overlay logo AFTER all drawings
    display_frame = overlay_logo_clipped(display_frame, logo, position="bottom-right")
    display_frame = overlay_logo_clipped(display_frame, qr_code, position="bottom-left")

    # Show frame
    cv2.imshow(WINDOW_NAME, display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    current_time = time.strftime('%Y%m%d-%H%M%S')
    filename_prefix = WINDOW_NAME.replace(' ', '_')

    if key == ord(' '):
        if not hasattr(cv2, "video_writer") or cv2.video_writer is None:
            video_filename = f"{filename_prefix}_{current_time}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            cv2.video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (display_frame.shape[1], display_frame.shape[0]))
            cv2.video_recording = True
            print(f"Started recording to {video_filename}")
        else:
            cv2.video_writer.release()
            cv2.video_writer = None
            cv2.video_recording = False
            print("Stopped video recording")

    if key == 32 and not hasattr(cv2, "last_space_time"):
        cv2.last_space_time = time.time()
    elif key != 32 and hasattr(cv2, "last_space_time"):
        still_filename = f"{filename_prefix}_{current_time}.jpg"
        cv2.imwrite(still_filename, display_frame)
        print(f"Screenshot saved as {still_filename}")
        del cv2.last_space_time  # No break here; loop continues

    # Write to video if recording
    if hasattr(cv2, "video_writer") and cv2.video_writer is not None:
        cv2.video_writer.write(display_frame)

cap.release()
cv2.destroyAllWindows()
