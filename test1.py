import cv2
import numpy as np
import time
import threading
import queue
from ultralytics import YOLO
import cvzone
import screeninfo
from pathlib import Path
import mediapipe as mp

# --- Configuration ---
CONFIG = {
    "rtsp_url": "rtsp://192.168.1.109:554/0/0/0",
    "webcam_id": 0,
    "use_webcam": True,
    "model_path": "models/yolo11n.pt", # Make sure you have a YOLO model here
    "logo_path": "img/odplogo.png",
    "qr_code_path": "img/qr-code.png",
    "window_name": "Multi-Tracking Demo",
    "info_text": "Attention: This demo uses live video only. No data is retained, stored or shared."
}

# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class MultiModelTrackerApp:
    def __init__(self, config):
        self.config = config
        self.stop_event = threading.Event()

        # --- Setup Queues ---
        self.frame_queue = queue.Queue(maxsize=2)
        self.yolo_results_queue = queue.Queue(maxsize=1)
        self.hand_results_queue = queue.Queue(maxsize=1)

        # --- Load Assets & Models ---
        self._setup_screen()
        self._load_assets()
        self.yolo_model = YOLO(self.config["model_path"])
        self.class_names = self.yolo_model.model.names
        
        # --- Video Source ---
        source = self.config["webcam_id"] if self.config["use_webcam"] else self.config["rtsp_url"]
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video stream: {source}")

        self.is_recording = False
        self.video_writer = None

    def _setup_screen(self):
        try:
            screen = screeninfo.get_monitors()[0]
            self.width, self.height = screen.width, screen.height
        except screeninfo.common.ScreenInfoError:
            print("Could not get screen info. Using 1280x720.")
            self.width, self.height = 1280, 720
        cv2.namedWindow(self.config["window_name"], cv2.WINDOW_AUTOSIZE)

    def _load_assets(self):
        self.logo = self._load_image(self.config["logo_path"])
        self.qr_code = self._load_image(self.config["qr_code_path"])

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Create a placeholder if image not found to avoid crashing
            print(f"Warning: Image not found at {path}. Creating a placeholder.")
            return np.zeros((100, 100, 4), dtype=np.uint8)
        return img

    def _frame_reader_thread(self):
        """Reads and resizes frames, then puts them into a queue."""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("End of stream or camera disconnected.")
                self.stop_event.set()
                break

            # Resize the frame immediately after capture
            resized_frame = cv2.resize(frame, (self.width, self.height))
            
            try:
                # Put the single resized frame into the queue for all consumers
                self.frame_queue.put(resized_frame, block=True, timeout=1)
            except queue.Full:
                continue
        print("Frame reader thread stopped.")

    def _yolo_processor_thread(self):
        """Processes frames with YOLO model."""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            # Run YOLO model
            results = self.yolo_model.track(frame, persist=True, classes=0, verbose=False)
            if results and results[0].boxes and results[0].boxes.id is not None:
                self.yolo_results_queue.put(results[0])

    def _hand_processor_thread(self):
        """Processes frames with MediaPipe Hands model."""
        with mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2) as hands:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Process with MediaPipe
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    self.hand_results_queue.put(results)
    
    def _is_high_five(self, hand_landmarks):
        """Checks for a high-five gesture (all non-thumb fingers extended)."""
        finger_tips = [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP
        ]
        pip_joints = [
            mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.PINKY_PIP
        ]
        
        for i in range(len(finger_tips)):
            tip = hand_landmarks.landmark[finger_tips[i]]
            pip = hand_landmarks.landmark[pip_joints[i]]
            # If any fingertip is below its middle joint, it's not a high five
            if tip.y > pip.y:
                return False
        return True

    def run(self):
        """Main application loop."""
        # --- Start all threads ---
        threads = [
            threading.Thread(target=self._frame_reader_thread),
            threading.Thread(target=self._yolo_processor_thread),
            threading.Thread(target=self._hand_processor_thread)
        ]
        for t in threads:
            t.daemon = True
            t.start()
        
        # Keep track of the last known results
        last_yolo_results = None
        last_hand_results = None

        while not self.stop_event.is_set():
            try:
                display_frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            display_frame = cv2.resize(frame, (self.width, self.height))

            # --- Get latest results without blocking ---
            try:
                last_yolo_results = self.yolo_results_queue.get_nowait()
            except queue.Empty:
                pass
            
            try:
                last_hand_results = self.hand_results_queue.get_nowait()
            except queue.Empty:
                pass
            
            # --- Draw Annotations ---
            high_five_active = False
            if last_hand_results:
                for hand_landmarks in last_hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    if self._is_high_five(hand_landmarks):
                        high_five_active = True
            
            if last_yolo_results:
                boxes = last_yolo_results.boxes.xyxy.int().cpu().tolist()
                track_ids = last_yolo_results.boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(display_frame, f"ID: {track_id}", (x1, y1 - 10), scale=1, thickness=1)

            # --- Draw Overlays ---
            if high_five_active:
                 cv2.putText(display_frame, "HIGH FIVE!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

            self._draw_info_text(display_frame)
            self._overlay_image(display_frame, self.logo, position="bottom-right")
            self._overlay_image(display_frame, self.qr_code, position="bottom-left")

            cv2.imshow(self.config["window_name"], display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_event.set()
                break
            
        self.cleanup(threads)
    
    # Other methods like _draw_info_text, _overlay_image, _toggle_recording etc. would be here
    # (Copied from the previous version for brevity)
    def _draw_info_text(self, frame):
        """Draws the informational text at the top of the screen."""
        text = self.config["info_text"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale, font_thickness = 1, 2
        text_size, _ = cv2.getTextSize(text, font, text_scale, font_thickness)
        
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = text_size[1] + 20
        
        fade = 0.5 * (1 + np.sin(time.time() * 2))
        color = (int(fade * 230), int(fade * 216), int(fade * 173)) # BGR
        
        cv2.putText(frame, text, (text_x, text_y), font, text_scale, color, font_thickness, cv2.LINE_AA)
        
    def _overlay_image(self, frame, overlay_img, position="bottom-right", margin=10):
        """Overlays a (potentially transparent) image on the frame."""
        fh, fw, _ = frame.shape
        # Check if overlay_img is valid
        if overlay_img is None or overlay_img.shape[0] == 0 or overlay_img.shape[1] == 0:
            return
            
        oh, ow, _ = overlay_img.shape
        max_width = fw // 5
        if ow > max_width:
            scale = max_width / ow
            overlay_img = cv2.resize(overlay_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            oh, ow, _ = overlay_img.shape
        
        if position == "bottom-right": x, y = fw - ow - margin, fh - oh - margin
        elif position == "bottom-left": x, y = margin, fh - oh - margin

        roi = frame[y:y+oh, x:x+ow]
        
        if overlay_img.shape[2] == 4:
            alpha = overlay_img[:, :, 3] / 255.0
            for c in range(0, 3):
                roi[:, :, c] = overlay_img[:, :, c] * alpha + roi[:, :, c] * (1.0 - alpha)
        else:
            roi[:] = overlay_img
            
        frame[y:y+oh, x:x+ow] = roi


    def cleanup(self, threads):
        """Waits for threads and releases resources."""
        print("Cleaning up resources...")
        for t in threads:
            t.join(timeout=2)
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    # Ensure asset directories exist
    Path("models").mkdir(exist_ok=True)
    Path("img").mkdir(exist_ok=True)
    
    try:
        app = MultiModelTrackerApp(CONFIG)
        app.run()
    except Exception as e:
        print(f"An error occurred: {e}")