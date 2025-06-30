import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
import cvzone
import screeninfo
from pathlib import Path

# --- Configuration ---
CONFIG = {
    "rtsp_url": "rtsp://192.168.1.109:554/0/0/0",
    "webcam_id": 0,
    "use_webcam": True,  # Set to False to use RTSP stream
    "model_path": "models/yolo11n.pt",
    "logo_path": "img/odplogo.png",
    "qr_code_path": "img/qr-code.png",
    "window_name": "People Tracker",
    "info_text": "Attention: This demo uses live video only. No data is retained, stored or shared."
}

# --- Frame Reader Thread ---
class FrameReader:
    """A threaded class to read frames from a video source."""
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise RuntimeError(f"Cannot open video stream: {src}")
        
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        """Starts the reading thread."""
        self.thread.start()
        return self

    def update(self):
        """Continuously reads frames from the stream."""
        while not self.stopped:
            self.ret, self.frame = self.stream.read()
            if not self.ret:
                self.stopped = True

    def read(self):
        """Returns the latest frame."""
        return self.frame

    def stop(self):
        """Stops the thread and releases resources."""
        self.stopped = True
        self.thread.join()
        self.stream.release()

# --- Main Application ---
class PeopleTrackerApp:
    """The main application class for the people tracker."""
    def __init__(self, config):
        self.config = config
        self._setup_screen()
        self._load_assets()
        
        self.model = YOLO(self.config["model_path"])
        self.class_names = self.model.model.names
        
        source = self.config["webcam_id"] if self.config["use_webcam"] else self.config["rtsp_url"]
        self.frame_reader = FrameReader(source).start()

        self.is_recording = False
        self.video_writer = None

    def _setup_screen(self):
        """Gets screen dimensions and creates a window."""
        try:
            screen = screeninfo.get_monitors()[0]
            self.width, self.height = screen.width, screen.height
        except screeninfo.common.ScreenInfoError:
            print("Could not get screen info. Using default 1280x720.")
            self.width, self.height = 1280, 720
        cv2.namedWindow(self.config["window_name"], cv2.WINDOW_AUTOSIZE)

    def _load_assets(self):
        """Loads images like logos and QR codes."""
        self.logo = self._load_image(self.config["logo_path"])
        self.qr_code = self._load_image(self.config["qr_code_path"])

    def _load_image(self, path):
        """Helper to load an image with an alpha channel."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found at: {path}")
        return img

    def run(self):
        """Main application loop."""
        frame_count = 0
        while not self.frame_reader.stopped:
            frame = self.frame_reader.read()
            if frame is None:
                continue

            frame_count += 1
            if frame_count % 3 != 0: # Process every 3rd frame
                continue

            # --- Processing ---
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            results = self.model.track(frame, persist=True, classes=0, verbose=False)
            
            # --- Drawing ---
            display_frame = frame.copy()
            if results and results[0].boxes and results[0].boxes.id is not None:
                self._draw_annotations(display_frame, results[0])
            
            self._draw_info_text(display_frame)
            self._overlay_image(display_frame, self.logo, position="bottom-right")
            self._overlay_image(display_frame, self.qr_code, position="bottom-left")

            # --- Display and Recording ---
            cv2.imshow(self.config["window_name"], display_frame)
            if self.is_recording and self.video_writer:
                self.video_writer.write(display_frame)

            # --- Input Handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._toggle_recording(display_frame)
            elif key == ord('s'):
                self._save_screenshot(display_frame)
        
        self.cleanup()

    def _draw_annotations(self, frame, results):
        """Draws bounding boxes and tracking IDs."""
        boxes = results.boxes.xyxy.int().cpu().tolist()
        track_ids = results.boxes.id.int().cpu().tolist()
        class_ids = results.boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = box
            class_name = self.class_names[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f"ID: {track_id}", (x1, y1 - 10), scale=1, thickness=1)

    def _draw_info_text(self, frame):
        """Draws the informational text at the top of the screen."""
        text = self.config["info_text"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale, font_thickness = 1, 2
        text_size, _ = cv2.getTextSize(text, font, text_scale, font_thickness)
        
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = text_size[1] + 20
        
        # Fading color effect
        fade = 0.5 * (1 + np.sin(time.time() * 2))
        color = (int(fade * 230), int(fade * 216), int(fade * 173)) # BGR
        
        cv2.putText(frame, text, (text_x, text_y), font, text_scale, color, font_thickness, cv2.LINE_AA)
        
    def _overlay_image(self, frame, overlay_img, position="bottom-right", margin=10):
        """Overlays a (potentially transparent) image on the frame."""
        fh, fw, _ = frame.shape
        oh, ow, _ = overlay_img.shape

        # Scale overlay if it's too large
        max_width = fw // 5
        if ow > max_width:
            scale = max_width / ow
            overlay_img = cv2.resize(overlay_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            oh, ow, _ = overlay_img.shape
        
        if position == "bottom-right":
            x, y = fw - ow - margin, fh - oh - margin
        elif position == "bottom-left":
            x, y = margin, fh - oh - margin
        # Add other positions as needed

        # Create a region of interest (ROI)
        roi = frame[y:y+oh, x:x+ow]
        
        # Blend using alpha channel if available
        if overlay_img.shape[2] == 4:
            alpha = overlay_img[:, :, 3] / 255.0
            for c in range(0, 3):
                roi[:, :, c] = overlay_img[:, :, c] * alpha + roi[:, :, c] * (1.0 - alpha)
        else:
            roi[:] = overlay_img
            
        frame[y:y+oh, x:x+ow] = roi

    def _toggle_recording(self, frame):
        """Starts or stops video recording."""
        self.is_recording = not self.is_recording
        if self.is_recording:
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            filename = f"{self.config['window_name'].replace(' ', '_')}_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w, _ = frame.shape
            self.video_writer = cv2.VideoWriter(filename, fourcc, 10.0, (w, h))
            print(f"Started recording to {filename}")
        else:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print("Stopped video recording.")

    def _save_screenshot(self, frame):
        """Saves a single frame as a JPG image."""
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        filename = f"Screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved as {filename}")

    def cleanup(self):
        """Releases all resources."""
        self.frame_reader.stop()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    Path("models").mkdir(exist_ok=True)
    Path("img").mkdir(exist_ok=True)
    
    try:
        app = PeopleTrackerApp(CONFIG)
        app.run()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")