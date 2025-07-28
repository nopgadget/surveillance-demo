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
from mediapipe.python.solutions import pose as mp_pose
import torch
import pytomlpp
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from mediapipe.framework.formats import landmark_pb2

SOURCE_RTSP = "rtsp"
SOURCE_WEBCAM = "webcam"
SOURCE_VIDEO = "video"

# --- Configuration ---
class Config:
    # all of the properties below are overwritten by config.json or example-config.json
    # these are the default values found in example-config.json
    # to modify these values, copy example-config.json to config.json and modify them there

    stream_source = SOURCE_RTSP # Change in your own config.toml
    """Possible values: rtsp, webcam, video"""

    rtsp_url = "rtsp://192.168.1.109:554/0/0/0" # default opencv stream
    webcam_id = 0
    video_path = "vid/hongdae.mp4" # Example video at https://www.youtube.com/watch?v=0qEczHL_Wlo

    fullscreen = False
    window_name_crowd = "People Counter"
    window_name_interactive = "Interactive"
    info_text = "No data is retained, stored or shared."
    info_text_interactive = "No data is retained, stored or shared. Use hand gestures for effects."

    logo_path = "img/odplogo.png"
    qr_code_path = "img/qr-code.png"
    face_overlay_path = "img/generic-face.png"  # Path to generic face image for overlay

    model_path = "models/yolo11n"
    """Model path shouldn't include extension, it will be automatically determined"""

    yolo_conf_threshold = 0.3 # Confidence threshold for YOLO detections

    def __init__(self, config_path = "config.toml"):
        if not os.path.exists(config_path):
            config_path = "example-config.toml"

        with open(config_path) as config_file:
            config_toml = pytomlpp.load(config_file)
        
        if not isinstance(config_toml, dict):
            return

        for k, v in config_toml.items():
            setattr(self, k, v)

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
    def __init__(self, config: Config):
        self.config: Config = config
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
        # Add FaceMesh queue for blackout
        self.face_results_queue = queue.Queue(maxsize=1)
        # Add Pose queue
        self.pose_results_queue = queue.Queue(maxsize=1)

        # --- Checkbox states for different features ---
        # Initialize with placeholder positions, will be updated after screen setup
        self.checkboxes = {
            # 'yolo_tracking': {'checked': True, 'rect': (0, 0, 20, 20), 'label': 'YOLO Tracking'},
            'hand_detection': {'checked': True, 'rect': (0, 0, 20, 20), 'label': 'Hand Detection'},
            'pose_detection': {'checked': False, 'rect': (0, 0, 20, 20), 'label': 'Pose Detection'},
            'ascii_effect': {'checked': False, 'rect': (0, 0, 20, 20), 'label': 'ASCII Effect'},
            'face_mesh': {'checked': False, 'rect': (0, 0, 20, 20), 'label': 'Face Mesh'},
            'face_overlay': {'checked': False, 'rect': (0, 0, 20, 20), 'label': 'Face Replace'},
            'face_blackout': {'checked': False, 'rect': (0, 0, 20, 20), 'label': 'Face Blackout'},
            'fps_counter': {'checked': True, 'rect': (0, 0, 20, 20), 'label': 'FPS Counter'},
            'info_display': {'checked': True, 'rect': (0, 0, 20, 20), 'label': 'Info & QR Code'},
            'second_window': {'checked': False, 'rect': (0, 0, 20, 20), 'label': 'Second Window'}
        }
        
        # Checkbox visibility state
        self.checkboxes_visible = False
        
        # Track previous second window state to handle hiding
        self.previous_second_window_state = False

        # --- Load Assets & Models ---
        self._setup_screen()
        self._load_assets()
        
        # --- Model selection logic: Prefer CUDA, then ONNX, then CPU ---
        onnx_path = self.config.model_path + ".onnx"
        pt_path = self.config.model_path + ".pt"
        if torch.cuda.favailable():
            print(f"CUDA is available. Using PyTorch model on CUDA: {pt_path}")
            self.yolo_model = YOLO(pt_path).to('cuda')
        elif os.path.exists(onnx_path):
            print(f"CUDA not available. Using ONNX model: {onnx_path}")
            self.yolo_model = YOLO(onnx_path, task="detect")
        else:
            print(f"CUDA not available and ONNX model not found. Using PyTorch model on CPU: {pt_path}")
            self.yolo_model = YOLO(pt_path).to('cpu')
        
        # --- Video Source ---
        source = None
        if self.config.stream_source == SOURCE_WEBCAM:
            source = self.config.webcam_id
        elif self.config.stream_source == SOURCE_VIDEO:
            source = self.config.video_path
        elif self.config.stream_source == SOURCE_RTSP:
            source = self.config.rtsp_url
        else:
            raise RuntimeError(f"Unsupported video source (use rtsp, webcam, or video): {self.config.stream_source}")
            
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video stream: {source}")

        # ASCII effect variables
        self.ascii_font_scale = 0.4       # Font scale for ASCII effect
        self.ascii_cell_size = 8          # Size of each ASCII cell
        self.ascii_chars = "@%#*+=-:. "   # Dark to light
        self.ascii_start_time = None       # Track when ASCII effect starts
        self.ascii_duration = 10.0        # Auto-disable ASCII after 10 seconds
        
        # Haptic text variables
        self.haptic_text = None
        self.haptic_text_start_time = None
        self.haptic_text_duration = 3.0  # Duration in seconds
        self.haptic_text_alpha = 0.0     # For fade effect
        
        # Gesture tracking variables
        self.gesture_start_time = None
        self.current_gesture = None
        self.gesture_duration = 2.0  # Duration to hold gesture
        self.gesture_cooldown = 3.0  # Cooldown after gesture activation (increased from 1.0)
        self.last_gesture_time = 0
        
        # Add FaceMesh instance for blackout
        self.face_mesh = None
        if mp_face_mesh is not None:
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        # Add FaceMesh instance for overlay
        self.face_mesh_overlay = None
        if mp_face_mesh is not None:
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

        # Video navigation controls (only for video files)
        self.video_paused = False
        self.is_video_file = self.config.stream_source == SOURCE_VIDEO

    def _toggle_video_pause(self):
        """Toggle video pause state (only for video files)."""
        if self.is_video_file:
            self.video_paused = not self.video_paused



    def _setup_screen(self):
        if self.config.fullscreen:
            try:
                screen = screeninfo.get_monitors()[0]
                self.width, self.height = screen.width, screen.height
                cv2.namedWindow(self.config.window_name_interactive, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(self.config.window_name_interactive, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except screeninfo.common.ScreenInfoError:
                print("Could not get screen info. Using 1280x720.")
                self.width, self.height = 1280, 720
                cv2.namedWindow(self.config.window_name_interactive, cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow(self.config.window_name_interactive, self.width, self.height)
        else:
            self.width, self.height = 1280, 720
            cv2.namedWindow(self.config.window_name_interactive, cv2.WINDOW_AUTOSIZE)
            cv2.resizeWindow(self.config.window_name_interactive, self.width, self.height)

        # Set mouse callback for checkbox interaction
        cv2.setMouseCallback(self.config.window_name_interactive, self._mouse_callback)
        
        # Update checkbox positions after screen dimensions are known
        self._update_checkbox_positions()

    def _update_checkbox_positions(self):
        """Update checkbox positions to be on the right side, vertically centered."""
        checkbox_width = 20
        checkbox_height = 20
        label_margin = 10
        checkbox_spacing = 40  # Vertical spacing between checkboxes
        
        # Calculate the right side position (with some margin from the edge)
        right_margin = 50
        checkbox_x = self.width - right_margin - checkbox_width - 150  # 150 pixels for label text
        
        # Calculate total height needed for all checkboxes
        total_checkboxes = len(self.checkboxes)
        total_height = (total_checkboxes - 1) * checkbox_spacing + checkbox_height
        
        # Calculate starting Y position to center vertically
        start_y = (self.height - total_height) // 2
        
        # Update each checkbox position
        for i, (feature_name, checkbox) in enumerate(self.checkboxes.items()):
            y_pos = start_y + i * checkbox_spacing
            checkbox['rect'] = (checkbox_x, y_pos, checkbox_width, checkbox_height)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for checkbox interaction."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is on QR code (bottom-left position)
            fh, fw = self.height, self.width
            qr_margin = 10
            qr_max_width = fw // 6
            
            # Calculate QR code position and size
            qr_oh, qr_ow = self.qr_code.shape[:2] if self.qr_code is not None else (100, 100)
            if qr_ow > qr_max_width:
                scale = qr_max_width / qr_ow
                qr_ow = int(qr_ow * scale)
                qr_oh = int(qr_oh * scale)
            
            qr_x, qr_y = qr_margin, fh - qr_oh - qr_margin
            
            # Check if click is within QR code bounds
            if (qr_x <= x <= qr_x + qr_ow and qr_y <= y <= qr_y + qr_oh):
                self.checkboxes_visible = not self.checkboxes_visible
                print(f"Checkboxes: {'visible' if self.checkboxes_visible else 'hidden'}")
            
            # Check checkbox clicks if checkboxes are visible
            elif self.checkboxes_visible:
                for feature_name, checkbox in self.checkboxes.items():
                    checkbox_x, checkbox_y, checkbox_w, checkbox_h = checkbox['rect']
                    if (checkbox_x <= x <= checkbox_x + checkbox_w and 
                        checkbox_y <= y <= checkbox_y + checkbox_h):
                        checkbox['checked'] = not checkbox['checked']
                        print(f"{checkbox['label']}: {'enabled' if checkbox['checked'] else 'disabled'}")

    def _draw_checkboxes(self, frame):
        """Draw all checkboxes on the frame."""
        if not self.checkboxes_visible:
            return
            
        for feature_name, checkbox in self.checkboxes.items():
            x, y, w, h = checkbox['rect']
            
            # Draw checkbox border
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            # Draw checkmark if checked
            if checkbox['checked']:
                # Draw a simple checkmark
                cv2.line(frame, (x + 3, y + h // 2), (x + w // 3, y + h - 3), (0, 255, 0), 2)
                cv2.line(frame, (x + w // 3, y + h - 3), (x + w - 3, y + 3), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(frame, checkbox['label'], (x + w + 10, y + h // 2 + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _load_assets(self):
        self.logo = self._load_image(self.config.logo_path)
        self.qr_code = self._load_image(self.config.qr_code_path)
        self.face_overlay = self._load_image(self.config.face_overlay_path)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Image not found at {path}. Creating a placeholder.")
            return np.zeros((100, 100, 4), dtype=np.uint8)
        return img

    def _frame_reader_thread(self):
        """Reads, resizes, and updates the shared self.latest_frame."""
        while not self.stop_event.is_set():
            # Check if video is paused (only for video files)
            if self.is_video_file and self.video_paused:
                time.sleep(0.01)  # Small delay when paused
                continue
                
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
                if not self.checkboxes['hand_detection']['checked']:
                    time.sleep(0.1)
                    continue
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

    def _count_fingers(self, hand_landmarks):
        """Count the number of extended fingers in a hand (excluding thumb)."""
        # Finger tip landmarks (excluding thumb)
        finger_tips = [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP
        ]
        
        # Corresponding PIP (second joint) landmarks
        finger_pips = [
            mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.PINKY_PIP
        ]
        
        extended_fingers = 0
        
        for i in range(len(finger_tips)):
            tip_y = hand_landmarks.landmark[finger_tips[i]].y
            pip_y = hand_landmarks.landmark[finger_pips[i]].y
            # Finger is extended if tip is above PIP joint
            if tip_y < pip_y - 0.01:  # Small threshold for stability
                extended_fingers += 1
        
        return extended_fingers

    def _handle_gesture_activation(self, finger_count):
        """Handle gesture-based feature activation (only enables features)."""
        current_time = time.time()
        
        # Check if we're in cooldown period
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return
        
        # Map finger count to features - each feature has its own hand position
        gesture_features = {
            1: 'pose_detection',      # 1 finger = pose detection
            2: 'ascii_effect',        # 2 fingers = ASCII effect
            3: 'face_mesh',           # 3 fingers = face mesh
            4: 'face_blackout',       # 4 fingers = face blackout
            5: 'face_overlay'         # 5 fingers = face overlay
        }
        
        if finger_count in gesture_features:
            feature_name = gesture_features[finger_count]
            
            # Only enable the feature if it's not already enabled
            if feature_name in self.checkboxes and not self.checkboxes[feature_name]['checked']:
                self.checkboxes[feature_name]['checked'] = True
                
                # Special handling for ASCII effect - start timer
                if feature_name == 'ascii_effect':
                    self.ascii_start_time = current_time
                    self._show_haptic_text(f"{self.checkboxes[feature_name]['label']} enabled for 10 seconds!")
                else:
                    self._show_haptic_text(f"{self.checkboxes[feature_name]['label']} enabled!")
                
                self.last_gesture_time = current_time
                print(f"Gesture {finger_count} fingers: {self.checkboxes[feature_name]['label']} enabled")
            else:
                # Feature is already enabled, show a different message
                self._show_haptic_text(f"{self.checkboxes[feature_name]['label']} is already enabled!")
                self.last_gesture_time = current_time
                print(f"Gesture {finger_count} fingers: {self.checkboxes[feature_name]['label']} already enabled")

    def _handle_gesture_deactivation(self):
        """Handle thumbs-down gesture to deactivate finger-controllable features."""
        current_time = time.time()
        
        # Check if we're in cooldown period
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return
        
        # Features that can be deactivated by thumbs down
        deactivatable_features = ['pose_detection', 'ascii_effect', 'face_mesh', 'face_blackout', 'face_overlay']
        
        deactivated_count = 0
        for feature_name in deactivatable_features:
            if feature_name in self.checkboxes and self.checkboxes[feature_name]['checked']:
                self.checkboxes[feature_name]['checked'] = False
                deactivated_count += 1
        
        if deactivated_count > 0:
            self._show_haptic_text(f"Deactivated {deactivated_count} features!")
            self.last_gesture_time = current_time
            print(f"Thumbs down: Deactivated {deactivated_count} features")

    def _draw_gesture_progress(self, frame, gesture, progress):
        """Draw gesture progress indicator."""
        if gesture is None or progress <= 0:
            return
            
        # Draw progress bar in top-left corner
        bar_width = 200
        bar_height = 20
        margin = 20
        
        # Background rectangle
        cv2.rectangle(frame, (margin, margin), (margin + bar_width, margin + bar_height), (50, 50, 50), -1)
        
        # Progress rectangle
        progress_width = int(bar_width * progress)
        if progress_width > 0:
            cv2.rectangle(frame, (margin, margin), (margin + progress_width, margin + bar_height), (0, 255, 0), -1)
        
        # Border
        cv2.rectangle(frame, (margin, margin), (margin + bar_width, margin + bar_height), (255, 255, 255), 2)
        
        # Text
        gesture_names = {1: "Pose", 2: "ASCII", 3: "Face Mesh", 4: "Face Blackout", 5: "Face Overlay"}
        text = f"{gesture_names.get(gesture, f'{gesture} Fingers')}: {int(progress * 100)}%"
        cv2.putText(frame, text, (margin + 5, margin + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_finger_count_debug(self, frame, finger_count):
        """Draw finger count for debugging."""
        if finger_count is None:
            return
            
        # Draw in top-right corner
        text = f"Fingers: {finger_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.7
        font_thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, text_scale, font_thickness)
        
        # Position in top-right
        margin = 20
        text_x = frame.shape[1] - text_width - margin
        text_y = text_height + margin
        
        # Background rectangle
        bg_padding = 10
        cv2.rectangle(frame, 
                     (text_x - bg_padding, text_y - text_height - bg_padding),
                     (text_x + text_width + bg_padding, text_y + bg_padding),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (text_x, text_y), font, text_scale, (255, 255, 255), font_thickness)

    def _draw_censored_box(self, frame, hand_landmarks):
        """Draw a black box with 'CENSORED' text over the middle finger gesture."""
        h, w = frame.shape[:2]
        
        # Get hand bounding box
        x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
        y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
        
        # Calculate bounding box with some padding
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add padding to make the box larger
        padding = 30
        min_x = max(0, min_x - padding)
        max_x = min(w, max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(h, max_y + padding)
        
        # Draw black rectangle
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 0), -1)
        
        # Add "CENSORED" text
        text = "CENSORED"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.8
        font_thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, text_scale, font_thickness)
        
        # Center text in the box
        text_x = min_x + (max_x - min_x - text_width) // 2
        text_y = min_y + (max_y - min_y + text_height) // 2
        
        # Draw text
        cv2.putText(frame, text, (text_x, text_y), font, text_scale, (255, 255, 255), font_thickness)



    def _is_thumbs_down(self, hand_landmarks):
        """Checks for a proper thumbs-down gesture."""
        # Get thumb landmarks
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        
        # For a proper thumbs down, we need:
        # 1. Thumb pointing downward (tip below IP joint)
        # 2. Other fingers curled in a fist
        # 3. Hand orientation should be more vertical than horizontal
        
        # Check that thumb is extended downward (tip below IP joint)
        if thumb_tip.y > thumb_ip.y:
            # Check that other fingers are curled (fist-like position)
            finger_tips = [
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP
            ]
            finger_pips = [
                mp_hands.HandLandmark.INDEX_FINGER_PIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                mp_hands.HandLandmark.RING_FINGER_PIP,
                mp_hands.HandLandmark.PINKY_PIP
            ]
            
            # Check if other fingers are curled (not extended)
            fingers_curled = True
            for i in range(len(finger_tips)):
                tip_y = hand_landmarks.landmark[finger_tips[i]].y
                pip_y = hand_landmarks.landmark[finger_pips[i]].y
                if tip_y < pip_y - 0.01:  # Finger is extended (with small threshold)
                    fingers_curled = False
                    break
            
            # Check hand orientation - for thumbs down, the hand should be more vertical
            # Compare wrist to finger positions to determine hand orientation
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            
            # Calculate hand width vs height
            hand_width = abs(index_tip.x - pinky_tip.x)
            hand_height = abs(thumb_tip.y - wrist.y)
            
            # Check that thumb is the lowest point of the hand
            # Get all finger tips to compare with thumb
            all_finger_tips = [
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            ]
            
            # Check if thumb is lower than all other finger tips
            thumb_is_lowest = True
            for finger_tip in all_finger_tips:
                if thumb_tip.y <= finger_tip.y:  # Thumb is not lower than this finger
                    thumb_is_lowest = False
                    break
            
            # For thumbs down, hand should be more vertical than horizontal
            # Also ensure thumb is pointing significantly downward and is the lowest point
            if (hand_height > hand_width * 0.8 and  # Hand is more vertical
                thumb_tip.y > wrist.y + 0.05 and    # Thumb is significantly below wrist
                thumb_is_lowest and                  # Thumb is the lowest point
                fingers_curled):
                return True
        
        return False

    def _is_middle_finger(self, hand_landmarks):
        """Checks for a middle finger gesture (flipping off)."""
        # Check if middle finger is extended while others are curled
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        
        # Middle finger should be extended (tip above PIP)
        if middle_tip.y >= middle_pip.y:
            return False
        
        # Check that other fingers (except thumb) are curled
        finger_tips = [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP
        ]
        finger_pips = [
            mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.PINKY_PIP
        ]
        
        # Check if other fingers are curled (not extended)
        for i in range(len(finger_tips)):
            tip_y = hand_landmarks.landmark[finger_tips[i]].y
            pip_y = hand_landmarks.landmark[finger_pips[i]].y
            if tip_y < pip_y:  # Finger is extended
                return False
        
        return True

    def _draw_info_text(self, frame, custom_text=None):
        """Draws the informational text at the top of the screen."""
        text = custom_text if custom_text is not None else self.config.info_text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale, font_thickness = 1, 2
        text_size, _ = cv2.getTextSize(text, font, text_scale, font_thickness)
        
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = text_size[1] + 20
        
        fade = 0.5 * (1 + np.sin(time.time() * 2))
        color = (int(fade * 230), int(fade * 216), int(fade * 173)) # BGR
        cv2.putText(frame, text, (text_x, text_y), font, text_scale, color, font_thickness, cv2.LINE_AA)

    def _draw_haptic_text(self, frame):
        """Draws haptic text messages on the left side of the screen."""
        if self.haptic_text is None or self.haptic_text_start_time is None:
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.haptic_text_start_time
        
        if elapsed_time > self.haptic_text_duration:
            # Clear the haptic text
            self.haptic_text = None
            self.haptic_text_start_time = None
            self.haptic_text_alpha = 0.0
            return
            
        # Calculate alpha for fade in/out effect
        if elapsed_time < 0.5:
            # Fade in
            self.haptic_text_alpha = elapsed_time / 0.5
        elif elapsed_time > self.haptic_text_duration - 0.5:
            # Fade out
            self.haptic_text_alpha = (self.haptic_text_duration - elapsed_time) / 0.5
        else:
            # Full opacity
            self.haptic_text_alpha = 1.0
            
        # Clamp alpha to 0-1
        self.haptic_text_alpha = max(0.0, min(1.0, self.haptic_text_alpha))
        
        # Draw the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.4  # Reduced from 1.2 to 0.24 (1/5 of original size)
        font_thickness = 1  # Reduced from 3 to 1 for smaller text
        
        # Calculate text size and position (left side, vertically centered)
        text_size, _ = cv2.getTextSize(self.haptic_text, font, text_scale, font_thickness)
        text_x = 50  # Left margin
        text_y = (frame.shape[0] + text_size[1]) // 2  # Vertically centered
        
        # Draw background rectangle for better visibility
        bg_padding = 20
        bg_alpha = self.haptic_text_alpha * 0.7  # Slightly transparent background
        bg_color = (int(0 * bg_alpha), int(0 * bg_alpha), int(0 * bg_alpha))  # Black background
        
        # Create background rectangle
        bg_x1 = text_x - bg_padding
        bg_y1 = text_y - text_size[1] - bg_padding
        bg_x2 = text_x + text_size[0] + bg_padding
        bg_y2 = text_y + bg_padding
        
        # Draw background with alpha blending
        roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
        if roi.size > 0:
            blended_bg = cv2.addWeighted(roi, 1 - bg_alpha, np.full_like(roi, bg_color), bg_alpha, 0)
            frame[bg_y1:bg_y2, bg_x1:bg_x2] = blended_bg
        
        # Draw text with alpha blending
        text_color = (0, 255, 0)  # Green text
        text_color_with_alpha = tuple(int(c * self.haptic_text_alpha) for c in text_color)
        
        cv2.putText(frame, self.haptic_text, (text_x, text_y), font, text_scale, 
                   text_color_with_alpha, font_thickness, cv2.LINE_AA)

    def _show_haptic_text(self, message):
        """Shows a haptic text message."""
        self.haptic_text = message
        self.haptic_text_start_time = time.time()
        self.haptic_text_alpha = 0.0
        
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
        """Convert a frame to an ASCII brightness-coded image with hacker green colors."""
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
                
                # Use hacker green color scheme based on brightness
                # Darker areas = darker green, brighter areas = brighter green
                green_intensity = int((avg / 255) * 255) + 100  # 0-255
                # Create a green color with some variation for a more authentic look
                color = (0, green_intensity, 0)  # BGR format: (Blue, Green, Red)
                
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
            # Only process if any face-mesh-dependent effect is enabled
            if not (self.checkboxes['face_mesh']['checked'] or self.checkboxes['face_overlay']['checked'] or self.checkboxes['face_blackout']['checked']):
                time.sleep(0.1)
                continue
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

    def _overlay_faces(self, frame, face_mesh_results):
        """Overlay a generic face image on detected faces using the convex hull of FaceMesh landmarks."""
        if face_mesh_results is None or not face_mesh_results.multi_face_landmarks:
            return frame
        
        if self.face_overlay is None:
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
                
                # Get the bounding rectangle of the face
                x, y, w_face, h_face = cv2.boundingRect(hull)
                
                # Resize the face overlay to match the detected face size, scaled up 2x
                face_overlay_resized = cv2.resize(self.face_overlay, (w_face * 2, h_face * 2))
                
                # Create a mask for the face region
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)
                
                # Extract the region of interest (accounting for 2x scaled overlay)
                # Center the larger overlay over the detected face
                x_offset = w_face // 2
                y_offset = h_face // 2
                roi_x = max(0, x - x_offset)
                roi_y = max(0, y - y_offset)
                roi_w = min(w_face * 2, frame.shape[1] - roi_x)
                roi_h = min(h_face * 2, frame.shape[0] - roi_y)
                
                roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                mask_roi = mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                
                if roi.shape[0] > 0 and roi.shape[1] > 0 and face_overlay_resized.shape[0] > 0 and face_overlay_resized.shape[1] > 0:
                    # Handle alpha channel if present
                    if face_overlay_resized.shape[2] == 4:
                        # Extract alpha channel and normalize
                        alpha = face_overlay_resized[:, :, 3] / 255.0
                        alpha = np.clip(alpha, 0, 1)
                        
                        # Blend the face overlay with the original frame
                        for c in range(3):
                            roi[:, :, c] = (face_overlay_resized[:, :, c] * alpha + 
                                           roi[:, :, c] * (1 - alpha)).astype(np.uint8)
                    else:
                        # No alpha channel, use the mask
                        mask_alpha = mask_roi / 255.0
                        for c in range(3):
                            roi[:, :, c] = (face_overlay_resized[:, :, c] * mask_alpha + 
                                           roi[:, :, c] * (1 - mask_alpha)).astype(np.uint8)
                    
                    # Put the blended region back
                    frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = roi
                    
        return frame

    def _blackout_faces(self, frame, face_mesh_results):
        """Blackout detected faces using the convex hull of FaceMesh landmarks."""
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
                
                # Create a black mask for the face region
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)
                
                # Apply black color to the face region
                black_color = (0, 0, 0)  # BGR format: black
                frame[mask > 0] = black_color
                    
        return frame

    def _pose_processor_thread(self):
        """Processes frames with MediaPipe Pose by accessing the shared frame."""
        while not self.stop_event.is_set():
            if not self.checkboxes['pose_detection']['checked']:
                time.sleep(0.1)
                continue
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
        # Add FaceMesh thread
        if self.face_mesh is not None:
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
        fps_x, fps_y = (self.width - self.width // 10, self.height - 40)
        frame_count_in_time_diff = 0.0
        new_frame_time = 0.0
        prev_frame_time = time.time()

        # Average FPS Initialization (end of playback)
        total_frame_count = 0
        start_frame_time = prev_frame_time

        while not self.stop_event.is_set():
            display_frame_interactive = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    display_frame_interactive = self.latest_frame.copy()
            
            if display_frame_interactive is None:
                time.sleep(0.01)
                continue

            display_frame_crowd = display_frame_interactive.copy()

            try:
                last_yolo_results = self.yolo_results_queue.get_nowait()
            except queue.Empty:
                pass
            
            try:
                last_hand_results = self.hand_results_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Get FaceMesh results
            if self.face_results_queue is not None:
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

            thumbs_down_detected = False
            middle_finger_detected = False
            current_gesture = None
            current_finger_count = None
            
            # --- Hand detection and gesture tracking ---
            if (self.checkboxes['hand_detection']['checked'] and 
                last_hand_results and last_hand_results.multi_hand_landmarks):
                for hand_landmarks in last_hand_results.multi_hand_landmarks:
                    # Check for middle finger gesture
                    if self._is_middle_finger(hand_landmarks):
                        middle_finger_detected = True
                        # Cancel any ongoing gesture detection when middle finger is detected
                        current_gesture = None
                        self.gesture_start_time = None
                        self.current_gesture = None
                    # Check for thumbs down gesture
                    elif self._is_thumbs_down(hand_landmarks):
                        thumbs_down_detected = True
                        # Cancel any ongoing gesture detection when thumbs-down is detected
                        current_gesture = None
                        self.gesture_start_time = None
                        self.current_gesture = None
                    else:
                        # Count fingers for gesture detection if not in special gesture positions
                        finger_count = self._count_fingers(hand_landmarks)
                        current_finger_count = finger_count
                        if 1 <= finger_count <= 5:
                            current_gesture = finger_count

            # --- Gesture duration logic ---
            now = time.time()
            if current_gesture is not None and not thumbs_down_detected and not middle_finger_detected:
                # Check if the gesture corresponds to an already enabled feature
                gesture_features = {
                    1: 'pose_detection',
                    2: 'ascii_effect', 
                    3: 'face_mesh',
                    4: 'face_blackout',
                    5: 'face_overlay'
                }
                
                feature_name = gesture_features.get(current_gesture)
                feature_already_enabled = (feature_name in self.checkboxes and 
                                        self.checkboxes[feature_name]['checked'])
                
                # Don't show progress for already enabled features
                if feature_already_enabled:
                    # Reset gesture tracking for already enabled features
                    self.gesture_start_time = None
                    self.current_gesture = None
                    current_gesture = None
                else:
                    # Normal gesture tracking for disabled features
                    if self.gesture_start_time is None or self.current_gesture != current_gesture:
                        self.gesture_start_time = now
                        self.current_gesture = current_gesture
                    elif now - self.gesture_start_time >= self.gesture_duration:
                        # Gesture held for required duration, trigger activation
                        self._handle_gesture_activation(current_gesture)
                        # Reset gesture tracking to stop progress bar
                        self.gesture_start_time = None
                        self.current_gesture = None
                        current_gesture = None  # Also reset the local variable
            elif thumbs_down_detected:
                # Handle thumbs-down deactivation
                self._handle_gesture_deactivation()
                # Reset gesture tracking after deactivation
                self.gesture_start_time = None
                self.current_gesture = None
                current_gesture = None
            else:
                # No gesture detected or special gestures detected, reset tracking
                self.gesture_start_time = None
                self.current_gesture = None
            # --- ASCII effect with auto-disable ---
            if (self.checkboxes['ascii_effect']['checked']):
                # Initialize ascii_start_time if it's None (checkbox was checked directly)
                if self.ascii_start_time is None:
                    self.ascii_start_time = now
                
                # Check if ASCII effect should auto-disable
                elapsed_time = now - self.ascii_start_time
                if elapsed_time >= self.ascii_duration:
                    # Auto-disable ASCII effect after duration
                    self.checkboxes['ascii_effect']['checked'] = False
                    self.ascii_start_time = None
                    self._show_haptic_text("ASCII effect auto-disabled!")
                    # Reset ASCII frame
                    with self.ascii_lock:
                        self.latest_ascii_frame = None
                else:
                    # Request ASCII conversion if not already requested
                    if not self.ascii_request_event.is_set():
                        self.ascii_request_event.set()
                    # Use the latest ASCII frame if available
                    with self.ascii_lock:
                        if self.latest_ascii_frame is not None:
                            display_frame_interactive = self.latest_ascii_frame.copy()
            else:
                # Reset ASCII frame and timer when not in ASCII mode
                self.ascii_start_time = None
                with self.ascii_lock:
                    self.latest_ascii_frame = None

            if last_yolo_results:
                boxes = last_yolo_results.boxes.xyxy.int().cpu().tolist()
                track_ids = last_yolo_results.boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    # Draw on crowd window
                    cv2.rectangle(display_frame_crowd, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cvzone.putTextRect(display_frame_crowd, f"{track_id}", (max(0, x1 + 10), max(35, y1 - 10)), scale=0.3, thickness=0, colorT=(0, 0, 0), colorR=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX)
                    # Draw on interactive window
                    cv2.rectangle(display_frame_interactive, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cvzone.putTextRect(display_frame_interactive, f"{track_id}", (max(0, x1 + 10), max(35, y1 - 10)), scale=0.3, thickness=0, colorT=(0, 0, 0), colorR=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX)

            # TODO: Re-enable this when we want to show the logo and QR code
            if self.checkboxes['info_display']['checked']:
                            self._draw_info_text(display_frame_interactive, self.config.info_text_interactive)
            self._overlay_image(display_frame_interactive, self.qr_code, position="bottom-left")
            self._draw_checkboxes(display_frame_interactive)  # Draw all checkboxes on the current frame

            # Info text and QR code always shown on crowd window
            self._draw_info_text(display_frame_crowd, self.config.info_text)
            self._overlay_image(display_frame_crowd, self.qr_code, position="bottom-left")

            # --- Draw video pause indicator on crowd window (only for video files) ---
            if self.is_video_file and self.video_paused:
                # Draw pause indicator in top-left corner
                pause_text = "PAUSED"
                text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                text_x, text_y = 20, 40
                
                # Draw background rectangle
                cv2.rectangle(display_frame_crowd, 
                            (text_x - 10, text_y - text_size[1] - 10),
                            (text_x + text_size[0] + 10, text_y + 10),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(display_frame_crowd, pause_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

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
            if self.checkboxes['fps_counter']['checked']:
                cv2.rectangle(display_frame_interactive, (fps_x - 5, fps_y + 5), (fps_x + fps_w + 5, fps_y - fps_h - 5), (0,0,0), -1)
                cv2.putText(display_frame_interactive, fps_string, (fps_x, fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # --- Overlay face (after ASCII effect) ---
            if (self.checkboxes['face_overlay']['checked']):
                display_frame_interactive = self._overlay_faces(display_frame_interactive, last_face_mesh_results)

            # --- Face blackout (before face mesh) ---
            if (self.checkboxes['face_blackout']['checked']):
                display_frame_interactive = self._blackout_faces(display_frame_interactive, last_face_mesh_results)

            # Run FaceMesh overlay processing on the original frame if enabled
            if self.checkboxes['face_mesh']['checked'] and self.face_mesh_overlay is not None:
                # Process on the original frame, not the ASCII frame
                with self.frame_lock:
                    if self.latest_frame is not None:
                        original_frame = self.latest_frame.copy()
                        img_rgb_overlay = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                        img_rgb_overlay.flags.writeable = False
                        last_face_mesh_overlay_results = self.face_mesh_overlay.process(img_rgb_overlay)

            # --- Face mesh overlay ---
            if (self.checkboxes['face_mesh']['checked']):
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
                            image=display_frame_interactive,
                            landmark_list=face_landmarks,
                            connections=tess,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=0),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(150,150,150), thickness=1, circle_radius=1)
                        )
                        mp_drawing.draw_landmarks(
                            image=display_frame_interactive,
                            landmark_list=face_landmarks,
                            connections=contours,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=1)
                        )
                        mp_drawing.draw_landmarks(
                            image=display_frame_interactive,
                            landmark_list=face_landmarks,
                            connections=irises,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=1)
                        )

            # --- Draw pose landmarks and connections ---
            if self.checkboxes['pose_detection']['checked'] and last_pose_results and last_pose_results.pose_landmarks:
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
                    image=display_frame_interactive,
                    landmark_list=pose_landmarks,
                    connections=pose_connections,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                )

            # --- Draw hand landmarks (after all effects are applied) ---
            orange_spec = mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=2, circle_radius=4)  # BGR: orange
            white_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4)
            if (self.checkboxes['hand_detection']['checked'] and 
                last_hand_results and last_hand_results.multi_hand_landmarks):
                for hand_landmarks in last_hand_results.multi_hand_landmarks:
                    if self._is_middle_finger(hand_landmarks):
                        # Draw black box with "CENSORED" text over the hand
                        self._draw_censored_box(display_frame_interactive, hand_landmarks)
                    elif self._is_thumbs_down(hand_landmarks):
                        mp_drawing.draw_landmarks(
                            display_frame_interactive, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            orange_spec, orange_spec
                        )
                    else:
                        mp_drawing.draw_landmarks(
                            display_frame_interactive, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            white_spec, white_spec
                        )

            # --- Draw gesture progress (before haptic text) ---
            if self.current_gesture is not None and self.gesture_start_time is not None:
                now = time.time()
                elapsed = now - self.gesture_start_time
                progress = min(elapsed / self.gesture_duration, 1.0)
                self._draw_gesture_progress(display_frame_interactive, self.current_gesture, progress)

            # --- Draw finger count debug (always show) ---
            self._draw_finger_count_debug(display_frame_interactive, current_finger_count)

            # --- Draw haptic text (on top of everything) ---
            self._draw_haptic_text(display_frame_interactive)

            # --- Draw video pause indicator (only for video files) ---
            if self.is_video_file and self.video_paused:
                # Draw pause indicator in top-left corner
                pause_text = "PAUSED"
                text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                text_x, text_y = 20, 40
                
                # Draw background rectangle
                cv2.rectangle(display_frame_interactive, 
                            (text_x - 10, text_y - text_size[1] - 10),
                            (text_x + text_size[0] + 10, text_y + 10),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(display_frame_interactive, pause_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow(self.config.window_name_interactive, display_frame_interactive)
            
            # Handle second window visibility
            current_second_window_state = self.checkboxes['second_window']['checked']
            if current_second_window_state:
                # Create the window if it doesn't exist
                if not self.previous_second_window_state:
                    if self.config.fullscreen:
                        cv2.namedWindow(self.config.window_name_crowd, cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty(self.config.window_name_crowd, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.namedWindow(self.config.window_name_crowd, cv2.WINDOW_AUTOSIZE)
                        cv2.resizeWindow(self.config.window_name_crowd, self.width, self.height)
                cv2.imshow(self.config.window_name_crowd, display_frame_crowd)
            elif self.previous_second_window_state and not current_second_window_state:
                # Hide the window when it was previously shown but now disabled
                cv2.destroyWindow(self.config.window_name_crowd)
            
            self.previous_second_window_state = current_second_window_state

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # q or escape key
                self.stop_event.set()
                break
            elif key == ord(' '):  # Space bar - toggle pause
                self._toggle_video_pause()

        
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
