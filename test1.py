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
import torch
import json
import os

# --- Configuration ---
class Config:
    # all of the properties below are overwritten by config.json or example-config.json
    # these are the default values found in example-config.json
    # to modify these values, copy example-config.json to config.json and modify them there

    rtsp_url = "rtsp://192.168.1.109:554/0/0/0" # default opencv stream

    use_webcam = False,
    webcam_id = 0,

    use_video_file = False, # Set to True to process a video file
    video_path = "vid/hongdae.mp4", # Example video at https://www.youtube.com/watch?v=0qEczHL_Wlo

    use_small_window = False,
    window_name = "Multi-Tracking Demo",
    info_text = "Live video processing only. No data is retained, stored or shared.",

    logo_path = "img/odplogo.png",
    qr_code_path = "img/qr-code.png",

    model_path = "models/yolo11n.pt", # Make sure you have a YOLO model file here
    yolo_conf_threshold = 0.3 # Confidence threshold for YOLO detections

    def __init__(self, config_path = "config.json"):
        if not os.path.exists(config_path):
            config_path = "example-config.json"

        with open(config_path, "r") as config_file:
            config_json = json.load(config_file)

        if not isinstance(config_json, dict):
            return

        for k, v in config_json.items():
            setattr(self, k, v)

# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class MultiModelTrackerApp:
    def __init__(self, config):
        self.config = config
        self.stop_event = threading.Event()

        # --- Shared frame variable and a lock ---
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # --- Queues are now only for results ---
        self.yolo_results_queue = queue.Queue(maxsize=1)
        self.hand_results_queue = queue.Queue(maxsize=1)

        # --- Load Assets & Models ---
        self._setup_screen()
        self._load_assets()
        
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.yolo_model = YOLO(self.config.model_path).to(self.device)
        
        # --- Video Source ---
        source = None
        if self.config.use_webcam:
            source = self.config.webcam_id
        elif self.config.use_video_file:
            source = self.config.video_path
        else:
            source = self.config.rtsp_url
            
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video stream: {source}")

        self.is_recording = False
        self.video_writer = None

    def _setup_screen(self):
        if self.config.use_small_window:
            try:
                screen = screeninfo.get_monitors()[0]
                self.width, self.height = screen.width, screen.height
            except screeninfo.common.ScreenInfoError:
                print("Could not get screen info. Using 1280x720.")
                self.width, self.height = 1280, 720

        else:
            self.width, self.height = 1280, 720

        cv2.namedWindow(self.config.window_name, cv2.WINDOW_AUTOSIZE)

    def _load_assets(self):
        self.logo = self._load_image(self.config.logo_path)
        self.qr_code = self._load_image(self.config.qr_code_path)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Image not found at {path}. Creating a placeholder.")
            return np.zeros((100, 100, 4), dtype=np.uint8)
        return img

    def _frame_reader_thread(self):
        """Reads, resizes, and updates the shared self.latest_frame."""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("End of stream or camera disconnected.")
                self.stop_event.set()
                break

            resized_frame = cv2.resize(frame, (self.width, self.height))

            with self.frame_lock:
                self.latest_frame = resized_frame.copy()
        print("Frame reader thread stopped.")

    def _yolo_processor_thread(self):
        """Processes frames with YOLO model by accessing the shared frame."""
        while not self.stop_event.is_set():
            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()

            if frame_to_process is not None:
                results = self.yolo_model.track(frame_to_process, persist=True, classes=0, verbose=False, conf=self.config.yolo_conf_threshold)
                if results and results[0].boxes and results[0].boxes.id is not None:
                    if not self.yolo_results_queue.empty():
                        try: self.yolo_results_queue.get_nowait()
                        except queue.Empty: pass
                    self.yolo_results_queue.put(results[0])
            
            time.sleep(0.02) # Prevent thread from running too fast

    def _hand_processor_thread(self):
        """Processes frames with MediaPipe by accessing the shared frame."""
        with mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=10) as hands:
            while not self.stop_event.is_set():
                frame_to_process = None
                with self.frame_lock:
                    if self.latest_frame is not None:
                        frame_to_process = self.latest_frame.copy()

                if frame_to_process is not None:
                    img_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                    img_rgb.flags.writeable = False
                    results = hands.process(img_rgb)
                    
                    if results.multi_hand_landmarks:
                        if not self.hand_results_queue.empty():
                            try: self.hand_results_queue.get_nowait()
                            except queue.Empty: pass
                        self.hand_results_queue.put(results)
                    else:
                        if not self.hand_results_queue.empty():
                            try: self.hand_results_queue.get_nowait()
                            except queue.Empty: pass
                        self.hand_results_queue.put(None)
                
                time.sleep(0.02) # Prevent thread from running too fast

    def _is_high_five(self, hand_landmarks):
        """Checks for a high-five gesture."""
        finger_tips_ids = [
            mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP
        ]
        pip_joints_ids = [
            mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP
        ]
        
        for i in range(len(finger_tips_ids)):
            if hand_landmarks.landmark[finger_tips_ids[i]].y > hand_landmarks.landmark[pip_joints_ids[i]].y:
                return False
        thumb_ip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
        index_finger_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
        
        x_proximity_threshold = 0.05 # This value might need tuning
        if abs(thumb_ip_x - index_finger_mcp_x) > x_proximity_threshold:
            return False
        return True

    def _draw_info_text(self, frame):
        """Draws the informational text at the top of the screen."""
        text = self.config.info_text
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
        if overlay_img is None or overlay_img.shape[0] == 0 or overlay_img.shape[1] == 0:
            return
            
        fh, fw, _ = frame.shape
        oh, ow, _ = overlay_img.shape
        max_width = fw // 6
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

    def run(self):
        """Main application loop."""
        threads = [
            threading.Thread(target=self._frame_reader_thread),
            threading.Thread(target=self._yolo_processor_thread),
            threading.Thread(target=self._hand_processor_thread)
        ]
        for t in threads:
            t.daemon = True
            t.start()
        
        last_yolo_results = None
        last_hand_results = None

        
        # FPS Counter Initialization
        TIME_DIFF_MAX = 0.5 # update FPS counter every X seconds
        fps_string = "FPS: N/A"
        fps_w, fps_h = cv2.getTextSize(fps_string, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        fps_x, fps_y = (self.width - self.width // 4, 80)
        frame_count_in_time_diff = 0.0
        new_frame_time = 0.0
        prev_frame_time = time.time()

        # Average FPS Initialization (end of playback)
        total_frame_count = 0
        start_frame_time = prev_frame_time

        while not self.stop_event.is_set():
            display_frame = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    display_frame = self.latest_frame.copy()
            
            if display_frame is None:
                time.sleep(0.01)
                continue

            try:
                last_yolo_results = self.yolo_results_queue.get_nowait()
            except queue.Empty:
                pass
            
            try:
                last_hand_results = self.hand_results_queue.get_nowait()
            except queue.Empty:
                pass
            
            high_five_count = 0
            # --- Green DrawingSpec for high five ---
            green_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)
            if last_hand_results and last_hand_results.multi_hand_landmarks:
                for hand_landmarks in last_hand_results.multi_hand_landmarks:
                    if self._is_high_five(hand_landmarks):
                        mp_drawing.draw_landmarks(
                            display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            green_spec, green_spec
                        )
                        high_five_count += 1
                    else:
                        mp_drawing.draw_landmarks(
                            display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

            if last_yolo_results:
                boxes = last_yolo_results.boxes.xyxy.int().cpu().tolist()
                track_ids = last_yolo_results.boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cvzone.putTextRect(display_frame, f"{track_id}", (max(0, x1 + 10), max(35, y1 - 10)), scale=0.3, thickness=0, colorT=(0, 0, 0), colorR=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX)

            if high_five_count > 0:
                text = f"{high_five_count} HIGH FIVE"
                text += "S!" if high_five_count > 1 else "!"
                cv2.putText(display_frame, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

            # TODO: Re-enable this when we want to show the logo and QR code
            #self._draw_info_text(display_frame)
            self._overlay_image(display_frame, self.logo, position="bottom-right")
            #self._overlay_image(display_frame, self.qr_code, position="bottom-left")

            # Average FPS Tally (end of playback)
            total_frame_count += 1

            # FPS Counter Calculation
            frame_count_in_time_diff += 1
            new_frame_time = time.time()
            time_diff = new_frame_time - prev_frame_time
            if time_diff >= TIME_DIFF_MAX:
                fps_string = "FPS: " + str(frame_count_in_time_diff // time_diff)
                frame_count_in_time_diff = 0
                prev_frame_time = new_frame_time

                fps_w, fps_h = cv2.getTextSize(fps_string, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # FPS Counter Display
            cv2.rectangle(display_frame, (fps_x - 5, fps_y + 5), (fps_x + fps_w + 5, fps_y - fps_h - 5), (0,0,0), -1)
            cv2.putText(display_frame, fps_string, (fps_x, fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(self.config.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # q or escape key
                self.stop_event.set()
                break
        
        # Average FPS Calcuation
        end_frame_time = time.time()
        time_diff = end_frame_time - start_frame_time
        if time_diff > 0:
            fps_string = "FPS: " + str(total_frame_count // time_diff)
        else:
            fps_string = "FPS: N/A"
        
        # Average FPS Display
        print("Average " + fps_string)

            
        self.cleanup(threads)

    def cleanup(self, threads):
        """Waits for threads and releases resources."""
        print("Cleaning up resources...")
        self.stop_event.set()
        for t in threads:
            t.join(timeout=2)
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    Path("img").mkdir(exist_ok=True)
    
    try:
        config = Config()
        app = MultiModelTrackerApp(config)
        app.run()
    except Exception as e:
        print(f"An error occurred: {e}")
