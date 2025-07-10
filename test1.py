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
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import pose as mp_pose
import torch
import json
import os
from mediapipe.framework.formats import landmark_pb2

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

    model_path = "models/yolo11n.pt",
    yolo_conf_threshold = 0.3 # Confidence threshold for YOLO detections
    ascii_on_high_five = True
    face_mesh_on_high_five = True

    def __init__(self, config_path = "config.json"):
        if not os.path.exists(config_path):
            config_path = "example-config.json"

        with open(config_path, "r") as config_file:
            config_json = json.load(config_file)

        if not isinstance(config_json, dict):
            return

        for k, v in config_json.items():
            setattr(self, k, v)
        # Ensure ascii_on_high_five is set (default True)
        if not hasattr(self, 'ascii_on_high_five'):
            self.ascii_on_high_five = True

# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
# mp_drawing, mp_drawing_styles, mp_pose already imported above
# Add FaceMesh
try:
    mp_face_mesh = __import__('mediapipe').solutions.face_mesh
except Exception as e:
    print("Warning: Could not import mediapipe face_mesh:", e)
    mp_face_mesh = None

class MultiModelTrackerApp:
    def __init__(self, config):
        self.config = config
        self.stop_event = threading.Event()

        # --- Shared frame variable and a lock ---
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # --- ASCII frame and lock ---
        self.latest_ascii_frame = None
        self.ascii_lock = threading.Lock()
        self.ascii_request_event = threading.Event()

        # --- Queues are now only for results ---
        self.yolo_results_queue = queue.Queue(maxsize=1)
        self.hand_results_queue = queue.Queue(maxsize=1)
        # Add FaceMesh queue if blackout is enabled
        self.face_results_queue = queue.Queue(maxsize=1) if getattr(self.config, 'blackout_face_on_high_five', False) else None
        # Add Pose queue
        self.pose_results_queue = queue.Queue(maxsize=1)

        # --- Load Assets & Models ---
        self._setup_screen()
        self._load_assets()
        
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # --- Model selection logic: Prefer CUDA, then ONNX, then CPU ---
        onnx_path = os.path.join('models', 'yolo11n.onnx')
        pt_path = self.config.model_path
        if torch.cuda.is_available():
            print(f"CUDA is available. Using PyTorch model on CUDA: {pt_path}")
            self.yolo_model = YOLO(pt_path).to('cuda')
        elif os.path.exists(onnx_path):
            print(f"CUDA not available. Using ONNX model: {onnx_path}")
            self.yolo_model = YOLO(onnx_path)
        else:
            print(f"CUDA not available and ONNX model not found. Using PyTorch model on CPU: {pt_path}")
            self.yolo_model = YOLO(pt_path).to('cpu')
        
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

        self.high_five_start_time = None  # Track when high five starts
        self.high_five_active = False     # Whether ASCII effect is active
        self.ascii_font_scale = 0.4       # Font scale for ASCII effect
        self.ascii_cell_size = 8          # Size of each ASCII cell
        self.ascii_chars = "@%#*+=-:. "   # Dark to light
        # Add FaceMesh instance if blackout is enabled
        self.face_mesh = None
        if getattr(self.config, 'blackout_face_on_high_five', False) and mp_face_mesh is not None:
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        # Add FaceMesh instance for overlay if enabled
        self.face_mesh_overlay = None
        if getattr(self.config, 'face_mesh_on_high_five', False) and mp_face_mesh is not None:
            self.face_mesh_overlay = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        # Add Pose instance
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

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

    def _ascii_processor_thread(self):
        """Converts frames to ASCII in a separate thread when requested."""
        while not self.stop_event.is_set():
            self.ascii_request_event.wait(timeout=0.1)
            if self.stop_event.is_set():
                break
            frame_to_ascii = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_ascii = self.latest_frame.copy()
            if frame_to_ascii is not None:
                ascii_frame = self._frame_to_ascii(frame_to_ascii)
                with self.ascii_lock:
                    self.latest_ascii_frame = ascii_frame
            self.ascii_request_event.clear()

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

    def _frame_to_ascii(self, frame):
        """Convert a frame to an ASCII brightness-coded image."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cell_size = self.ascii_cell_size
        chars = self.ascii_chars
        n_chars = len(chars)
        out_img = np.zeros_like(frame)
        for y in range(0, h, cell_size):
            for x in range(0, w, cell_size):
                cell = gray[y:y+cell_size, x:x+cell_size]
                if cell.size == 0:
                    continue
                avg = int(np.mean(cell))
                char_idx = int((avg / 255) * (n_chars - 1))
                char = chars[char_idx]
                # Use grayscale for the text color
                color = (int(avg), int(avg), int(avg))
                cv2.putText(
                    out_img, char, (x, y + cell_size),
                    cv2.FONT_HERSHEY_SIMPLEX, self.ascii_font_scale, color, 1, cv2.LINE_AA
                )
        return out_img

    def _face_mesh_processor_thread(self):
        """Processes frames with MediaPipe FaceMesh by accessing the shared frame."""
        if self.face_mesh is None or self.face_results_queue is None:
            return
        while not self.stop_event.is_set():
            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            if frame_to_process is not None:
                img_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = self.face_mesh.process(img_rgb)
                if not self.face_results_queue.empty():
                    try: self.face_results_queue.get_nowait()
                    except queue.Empty: pass
                self.face_results_queue.put(results)
            time.sleep(0.02)

    def _blackout_faces(self, frame, face_mesh_results):
        """Black out faces in the frame using the convex hull of FaceMesh landmarks."""
        if face_mesh_results is None or not face_mesh_results.multi_face_landmarks:
            return frame
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            points = []
            h, w, _ = frame.shape
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                points.append([x, y])
            points = np.array(points, dtype=np.int32)
            if points.shape[0] > 0:
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(frame, hull, (0, 0, 0))
        return frame

    def _pose_processor_thread(self):
        """Processes frames with MediaPipe Pose by accessing the shared frame."""
        while not self.stop_event.is_set():
            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            if frame_to_process is not None:
                img_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = self.pose.process(img_rgb)
                if not self.pose_results_queue.empty():
                    try: self.pose_results_queue.get_nowait()
                    except queue.Empty: pass
                self.pose_results_queue.put(results)
            time.sleep(0.02)

    def run(self):
        """Main application loop."""
        threads = [
            threading.Thread(target=self._frame_reader_thread),
            threading.Thread(target=self._yolo_processor_thread),
            threading.Thread(target=self._hand_processor_thread),
            threading.Thread(target=self._ascii_processor_thread)
        ]
        # Add FaceMesh thread if blackout is enabled
        if getattr(self.config, 'blackout_face_on_high_five', False) and self.face_mesh is not None:
            threads.append(threading.Thread(target=self._face_mesh_processor_thread))
        # Add Pose thread
        threads.append(threading.Thread(target=self._pose_processor_thread))
        for t in threads:
            t.daemon = True
            t.start()
        
        last_yolo_results = None
        last_hand_results = None
        last_face_mesh_results = None
        last_face_mesh_overlay_results = None
        last_pose_results = None

        
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
            
            # Get FaceMesh results if blackout is enabled
            if getattr(self.config, 'blackout_face_on_high_five', False) and self.face_results_queue is not None:
                try:
                    last_face_mesh_results = self.face_results_queue.get_nowait()
                except queue.Empty:
                    pass

            # Get Pose results
            if self.pose_results_queue is not None:
                try:
                    last_pose_results = self.pose_results_queue.get_nowait()
                except queue.Empty:
                    pass

            high_five_count = 0
            high_five_detected = False
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
                        high_five_detected = True
                    else:
                        mp_drawing.draw_landmarks(
                            display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

            # --- High five duration logic ---
            now = time.time()
            if high_five_detected:
                if self.high_five_start_time is None:
                    self.high_five_start_time = now
                elif now - self.high_five_start_time >= 1.0:  # 1 second for face mesh overlay
                    self.high_five_active = True
            else:
                self.high_five_start_time = None
                self.high_five_active = False
            # --- ASCII effect ---
            if self.high_five_active and self.config.ascii_on_high_five:
                # Request ASCII conversion if not already requested
                if not self.ascii_request_event.is_set():
                    self.ascii_request_event.set()
                # Use the latest ASCII frame if available
                with self.ascii_lock:
                    if self.latest_ascii_frame is not None:
                        display_frame = self.latest_ascii_frame.copy()
            else:
                # Reset ASCII frame when not in ASCII mode
                with self.ascii_lock:
                    self.latest_ascii_frame = None
            # --- Blackout face on high five ---
            if self.high_five_active and getattr(self.config, 'blackout_face_on_high_five', False):
                display_frame = self._blackout_faces(display_frame, last_face_mesh_results)

            if last_yolo_results:
                boxes = last_yolo_results.boxes.xyxy.int().cpu().tolist()
                track_ids = last_yolo_results.boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cvzone.putTextRect(display_frame, f"{track_id}", (max(0, x1 + 10), max(35, y1 - 10)), scale=0.3, thickness=0, colorT=(0, 0, 0), colorR=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX)

            # TODO: Re-enable this when we want to show the logo and QR code
            self._draw_info_text(display_frame)
            #self._overlay_image(display_frame, self.logo, position="bottom-right")
            self._overlay_image(display_frame, self.qr_code, position="bottom-left")

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

            # Run FaceMesh overlay processing on the current frame if enabled
            if self.high_five_active and getattr(self.config, 'face_mesh_on_high_five', False) and self.face_mesh_overlay is not None:
                img_rgb_overlay = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img_rgb_overlay.flags.writeable = False
                last_face_mesh_overlay_results = self.face_mesh_overlay.process(img_rgb_overlay)

            # --- Face mesh overlay on high five ---
            if self.high_five_active and getattr(self.config, 'face_mesh_on_high_five', False):
                if (
                    mp_face_mesh is not None and
                    last_face_mesh_overlay_results and
                    hasattr(last_face_mesh_overlay_results, 'multi_face_landmarks') and
                    last_face_mesh_overlay_results.multi_face_landmarks is not None and
                    isinstance(last_face_mesh_overlay_results.multi_face_landmarks, (list, tuple))
                ):
                    def filter_connections(connections, num_landmarks):
                        return [conn for conn in connections if max(conn) < num_landmarks]
                    for face_landmarks in list(last_face_mesh_overlay_results.multi_face_landmarks):
                        num_landmarks = len(face_landmarks.landmark)
                        tess = filter_connections(mp_face_mesh.FACEMESH_TESSELATION, num_landmarks)
                        contours = filter_connections(mp_face_mesh.FACEMESH_CONTOURS, num_landmarks)
                        irises = filter_connections(mp_face_mesh.FACEMESH_IRISES, num_landmarks)
                        mp_drawing.draw_landmarks(
                            image=display_frame,
                            landmark_list=face_landmarks,
                            connections=tess,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                        )
                        mp_drawing.draw_landmarks(
                            image=display_frame,
                            landmark_list=face_landmarks,
                            connections=contours,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1)
                        )
                        mp_drawing.draw_landmarks(
                            image=display_frame,
                            landmark_list=face_landmarks,
                            connections=irises,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1)
                        )

            # --- Draw pose landmarks and connections ---
            if last_pose_results and last_pose_results.pose_landmarks:
                # Draw up to the wrists (shoulders, elbows, wrists, torso, legs), exclude hand/finger landmarks (17-22)
                pose_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
                index_map = {orig_idx: i for i, orig_idx in enumerate(pose_indices)}
                # Filter connections to only those with both indices in pose_indices
                pose_connections = [
                    (index_map[a], index_map[b])
                    for (a, b) in list(mp_pose.POSE_CONNECTIONS)
                    if a in pose_indices and b in pose_indices
                ]
                NormalizedLandmarkList = getattr(landmark_pb2, 'NormalizedLandmarkList')
                all_landmarks = last_pose_results.pose_landmarks.landmark
                pose_landmarks = NormalizedLandmarkList(
                    landmark=[all_landmarks[i] for i in pose_indices]
                )
                mp_drawing.draw_landmarks(
                    image=display_frame,
                    landmark_list=pose_landmarks,
                    connections=pose_connections,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                )

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
