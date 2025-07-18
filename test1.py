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
import pytomlpp
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from mediapipe.framework.formats import landmark_pb2
import dlib

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
    info_text_interactive = "No data is retained, stored or shared. Hold up hand for additional effects."

    logo_path = "img/odplogo.png"
    qr_code_path = "img/qr-code.png"
    face_overlay_path = "img/generic-face.png"  # Path to generic face image for overlay
    face_swap_source_path = "img/musk.jpg"  # Path to source face for face swap

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
            'pose_detection': {'checked': True, 'rect': (0, 0, 20, 20), 'label': 'Pose Detection'},
            'ascii_effect': {'checked': True, 'rect': (0, 0, 20, 20), 'label': 'ASCII Effect'},
            'face_mesh': {'checked': True, 'rect': (0, 0, 20, 20), 'label': 'Face Mesh'},
            'face_overlay': {'checked': False, 'rect': (0, 0, 20, 20), 'label': 'Face Replace'},
            'face_blackout': {'checked': True, 'rect': (0, 0, 20, 20), 'label': 'Face Blackout'},
            'face_swap': {'checked': False, 'rect': (0, 0, 20, 20), 'label': 'Face Swap'},
            'fps_counter': {'checked': True, 'rect': (0, 0, 20, 20), 'label': 'FPS Counter'},
            'info_display': {'checked': True, 'rect': (0, 0, 20, 20), 'label': 'Info & QR Code'},
            'second_window': {'checked': False, 'rect': (0, 0, 20, 20), 'label': 'Second Window'}
        }
        
        # Checkbox visibility state
        self.checkboxes_visible = True
        
        # Track previous second window state to handle hiding
        self.previous_second_window_state = True

        # --- Load Assets & Models ---
        self._setup_screen()
        self._load_assets()
        
        # --- Model selection logic: Prefer CUDA, then ONNX, then CPU ---
        onnx_path = self.config.model_path + ".onnx"
        pt_path = self.config.model_path + ".pt"
        if torch.cuda.is_available():
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

        self.is_recording = False
        self.video_writer = None

        self.high_five_start_time = None  # Track when high five starts
        self.high_five_active = False     # Whether ASCII effect is active
        self.ascii_font_scale = 0.4       # Font scale for ASCII effect
        self.ascii_cell_size = 8          # Size of each ASCII cell
        self.ascii_chars = "@%#*+=-:. "   # Dark to light
        
        # Haptic text variables
        self.haptic_text = None
        self.haptic_text_start_time = None
        self.haptic_text_duration = 3.0  # Duration in seconds
        self.haptic_text_alpha = 0.0     # For fade effect
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

        # Initialize face swap components
        self._setup_face_swap()

    def _setup_screen(self):
        if self.config.fullscreen:
            try:
                screen = screeninfo.get_monitors()[0]
                self.width, self.height = screen.width, screen.height
                cv2.namedWindow(self.config.window_name_crowd, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(self.config.window_name_crowd, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.namedWindow(self.config.window_name_interactive, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(self.config.window_name_interactive, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except screeninfo.common.ScreenInfoError:
                print("Could not get screen info. Using 1280x720.")
                self.width, self.height = 1280, 720
                cv2.namedWindow(self.config.window_name_crowd, cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow(self.config.window_name_crowd, self.width, self.height)
                cv2.namedWindow(self.config.window_name_interactive, cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow(self.config.window_name_interactive, self.width, self.height)
        else:
            self.width, self.height = 1280, 720
            cv2.namedWindow(self.config.window_name_crowd, cv2.WINDOW_AUTOSIZE)
            cv2.resizeWindow(self.config.window_name_crowd, self.width, self.height)
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

    def _setup_face_swap(self):
        """Initialize face swap components."""
        try:
            self.face_swap_source = cv2.imread(self.config.face_swap_source_path)
            if self.face_swap_source is None:
                print(f"Warning: Face swap source image not found at {self.config.face_swap_source_path}")
                self.face_swap_available = False
                return

            self.face_swap_source_gray = cv2.cvtColor(self.face_swap_source, cv2.COLOR_BGR2GRAY)
            self.face_swap_mask = np.zeros_like(self.face_swap_source_gray)

            # Initialize dlib face detector and predictor
            self.face_detector = dlib.get_frontal_face_detector()
            self.face_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

            # Detect faces in source image
            faces = self.face_detector(self.face_swap_source_gray, 1)
            if len(faces) == 0:
                print("No faces detected in face swap source image")
                self.face_swap_available = False
                return

            face = faces[0]
            landmarks = self.face_predictor(self.face_swap_source_gray, face)
            self.face_swap_landmark_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                self.face_swap_landmark_points.append((x, y))

            points = np.array(self.face_swap_landmark_points, np.int32)
            convexhull = cv2.convexHull(points)
            cv2.fillConvexPoly(self.face_swap_mask, convexhull, (255,))

            self.face_swap_face_image = cv2.bitwise_and(self.face_swap_source, self.face_swap_source, mask=self.face_swap_mask)

            rect = cv2.boundingRect(convexhull)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(self.face_swap_landmark_points)
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)

            self.face_swap_indexes_triangles = []
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                index_pt1 = np.where((points == pt1).all(axis=1))
                index_pt1 = self._extract_index_nparray(index_pt1)

                index_pt2 = np.where((points == pt2).all(axis=1))
                index_pt2 = self._extract_index_nparray(index_pt2)
                
                index_pt3 = np.where((points == pt3).all(axis=1))
                index_pt3 = self._extract_index_nparray(index_pt3)

                if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                    triangle = [index_pt1, index_pt2, index_pt3]
                    self.face_swap_indexes_triangles.append(triangle)

            self.face_swap_available = True
            print("Face swap initialized successfully")
        except Exception as e:
            print(f"Error initializing face swap: {e}")
            self.face_swap_available = False
            # Set default values to prevent crashes
            self.face_swap_source = None
            self.face_swap_source_gray = None
            self.face_swap_mask = None
            self.face_swap_face_image = None
            self.face_swap_landmark_points = []
            self.face_swap_indexes_triangles = []

    def _extract_index_nparray(self, nparray):
        """Extract index from numpy array."""
        index = None
        for num in nparray[0]:
            index = num
            break
        return index

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

    def _apply_face_swap(self, frame):
        """Apply face swap to the frame."""
        if not self.face_swap_available or not hasattr(self, 'face_swap_indexes_triangles') or self.face_swap_source is None:
            return frame

        img2_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img2_new_face = np.zeros_like(frame)
        
        faces2 = self.face_detector(img2_gray, 1)
        
        if len(faces2) > 0:
            for face2 in faces2:
                img2_new_face = np.zeros_like(frame)
                
                landmarks2 = self.face_predictor(img2_gray, face2)
                landmark_points2 = []
                for n in range(0, 68):
                    x = landmarks2.part(n).x
                    y = landmarks2.part(n).y
                    landmark_points2.append((x, y))

                img2_face_mask = np.zeros_like(img2_gray)
                points2 = np.array(landmark_points2, np.int32)
                convexhull2 = cv2.convexHull(points2)
                cv2.fillConvexPoly(img2_face_mask, convexhull2, (255,))

                for triangle_index in self.face_swap_indexes_triangles:
                    tr1_pt1 = self.face_swap_landmark_points[triangle_index[0]]
                    tr1_pt2 = self.face_swap_landmark_points[triangle_index[1]]
                    tr1_pt3 = self.face_swap_landmark_points[triangle_index[2]]
                    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                    rect1 = cv2.boundingRect(triangle1)
                    (x, y, w, h) = rect1
                    cropped_triangle = self.face_swap_source[y: y + h, x: x + w]
                    cropped_tr1_mask = np.zeros((h, w), np.uint8)

                    points = np.array([
                        [tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]
                    ], np.int32)

                    cv2.fillConvexPoly(cropped_tr1_mask, points, (255,))
                    cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)

                    tr2_pt1 = landmark_points2[triangle_index[0]]
                    tr2_pt2 = landmark_points2[triangle_index[1]]
                    tr2_pt3 = landmark_points2[triangle_index[2]]
                    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

                    rect2 = cv2.boundingRect(triangle2)
                    (x, y, w, h) = rect2
                    cropped_triangle2 = frame[y: y + h, x: x + w]

                    cropped_tr2_mask = np.zeros((h, w), np.uint8)

                    points2 = np.array([
                        [tr2_pt1[0] - x, tr2_pt1[1] - y],
                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                        [tr2_pt3[0] - x, tr2_pt3[1] - y]
                    ], np.int32)

                    cv2.fillConvexPoly(cropped_tr2_mask, points2, (255,))
                    cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask=cropped_tr2_mask)

                    src_points = points.astype(np.float32)
                    dst_points = points2.astype(np.float32)
                    
                    if src_points.shape[0] == 3 and dst_points.shape[0] == 3:
                        try:
                            M = cv2.getAffineTransform(src_points, dst_points)
                            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                            
                            warped_mask = cv2.warpAffine(cropped_tr1_mask, M, (w, h))
                            
                            triangle_area = img2_new_face[y:y + h, x:x + w]
                            
                            warped_mask_3d = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
                            warped_mask_3d = warped_mask_3d.astype(np.float32) / 255.0
                            
                            alpha = warped_mask_3d
                            beta = 1.0 - alpha
                            blended = cv2.addWeighted(triangle_area, 1.0, warped_triangle, 1.0, 0)
                            blended = blended * beta + warped_triangle * alpha
                            img2_new_face[y:y + h, x:x + w] = blended.astype(np.uint8)
                        except cv2.error:
                            continue

                img2_new_face_gray = cv2.cvtColor(img2_new_face, cv2.COLOR_BGR2GRAY)
                _, face_mask = cv2.threshold(img2_new_face_gray, 1, 255, cv2.THRESH_BINARY)

                face_mask = cv2.GaussianBlur(face_mask, (3, 3), 0)
                face_mask = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2BGR)
                face_mask = face_mask.astype(np.float32) / 255.0

                background = frame.astype(np.float32)
                new_face = img2_new_face.astype(np.float32)

                result = background * (1.0 - face_mask) + new_face * face_mask
                result = result.astype(np.uint8)

                face_mask_gray = cv2.cvtColor(face_mask.astype(np.uint8) * 255, cv2.COLOR_BGR2GRAY)
                _, face_mask_binary = cv2.threshold(face_mask_gray, 127, 255, cv2.THRESH_BINARY)

                moments = cv2.moments(face_mask_binary)
                if moments["m00"] != 0:
                    center_x = int(moments["m10"] / moments["m00"])
                    center_y = int(moments["m01"] / moments["m00"])
                    center = (center_x, center_y)
                    
                    result = cv2.seamlessClone(img2_new_face, frame, face_mask_binary, center, cv2.NORMAL_CLONE)
                
                frame = result
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

            high_five_count = 0
            high_five_detected = False
            # --- Green DrawingSpec for high five ---
            if (self.checkboxes['hand_detection']['checked'] and 
                last_hand_results and last_hand_results.multi_hand_landmarks):
                for hand_landmarks in last_hand_results.multi_hand_landmarks:
                    if self._is_high_five(hand_landmarks):
                        high_five_count += 1
                        high_five_detected = True

            # --- High five duration logic ---
            now = time.time()
            if high_five_detected:
                if self.high_five_start_time is None:
                    self.high_five_start_time = now
                elif now - self.high_five_start_time >= 1.0:  # 1 second for face mesh overlay
                    if not self.high_five_active:
                        self.high_five_active = True
                        self._show_haptic_text("Effects enabled due to high five!")
            else:
                if self.high_five_start_time is not None:
                    self.high_five_start_time = None
                    if self.high_five_active:
                        self.high_five_active = False
                        self._show_haptic_text("Effects disabled due to no high five")
                    else:
                        self.high_five_start_time = None
                        self.high_five_active = False
            # --- ASCII effect ---
            if (self.high_five_active and 
                self.checkboxes['ascii_effect']['checked']):
                # Request ASCII conversion if not already requested
                if not self.ascii_request_event.is_set():
                    self.ascii_request_event.set()
                # Use the latest ASCII frame if available
                with self.ascii_lock:
                    if self.latest_ascii_frame is not None:
                        display_frame_interactive = self.latest_ascii_frame.copy()
            else:
                # Reset ASCII frame when not in ASCII mode
                with self.ascii_lock:
                    self.latest_ascii_frame = None

            if last_yolo_results:
                boxes = last_yolo_results.boxes.xyxy.int().cpu().tolist()
                track_ids = last_yolo_results.boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display_frame_crowd, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cvzone.putTextRect(display_frame_crowd, f"{track_id}", (max(0, x1 + 10), max(35, y1 - 10)), scale=0.3, thickness=0, colorT=(0, 0, 0), colorR=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX)

            # TODO: Re-enable this when we want to show the logo and QR code
            if self.checkboxes['info_display']['checked']:
                self._draw_info_text(display_frame_interactive, self.config.info_text_interactive)
                #self._overlay_image(display_frame, self.logo, position="bottom-right")
                self._overlay_image(display_frame_interactive, self.qr_code, position="bottom-left")
            self._draw_checkboxes(display_frame_interactive)  # Draw all checkboxes on the current frame

            # Info text and QR code always shown on crowd window
            self._draw_info_text(display_frame_crowd, self.config.info_text)
            #self._overlay_image(display_frame, self.logo, position="bottom-right")
            self._overlay_image(display_frame_crowd, self.qr_code, position="bottom-left")

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

            # --- Overlay face on high five (after ASCII effect) ---
            if (self.high_five_active and 
                self.checkboxes['face_overlay']['checked']):
                display_frame_interactive = self._overlay_faces(display_frame_interactive, last_face_mesh_results)

            # --- Face blackout on high five (before face mesh) ---
            if (self.high_five_active and 
                self.checkboxes['face_blackout']['checked']):
                display_frame_interactive = self._blackout_faces(display_frame_interactive, last_face_mesh_results)

            # --- Face swap (independent of high five) ---
            if self.checkboxes['face_swap']['checked']:
                display_frame_interactive = self._apply_face_swap(display_frame_interactive)

            # Run FaceMesh overlay processing on the original frame if enabled
            if self.high_five_active and self.checkboxes['face_mesh']['checked'] and self.face_mesh_overlay is not None:
                # Process on the original frame, not the ASCII frame
                with self.frame_lock:
                    if self.latest_frame is not None:
                        original_frame = self.latest_frame.copy()
                        img_rgb_overlay = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                        img_rgb_overlay.flags.writeable = False
                        last_face_mesh_overlay_results = self.face_mesh_overlay.process(img_rgb_overlay)

            # --- Face mesh overlay on high five ---
            if (self.high_five_active and 
                self.checkboxes['face_mesh']['checked']):
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
            green_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)
            white_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4)
            if (self.checkboxes['hand_detection']['checked'] and 
                last_hand_results and last_hand_results.multi_hand_landmarks):
                for hand_landmarks in last_hand_results.multi_hand_landmarks:
                    if self._is_high_five(hand_landmarks):
                        mp_drawing.draw_landmarks(
                            display_frame_interactive, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            green_spec, green_spec
                        )
                    else:
                        mp_drawing.draw_landmarks(
                            display_frame_interactive, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            white_spec, white_spec
                        )

            # --- Draw haptic text (on top of everything) ---
            self._draw_haptic_text(display_frame_interactive)

            cv2.imshow(self.config.window_name_interactive, display_frame_interactive)
            
            # Handle second window visibility
            current_second_window_state = self.checkboxes['second_window']['checked']
            if current_second_window_state:
                cv2.imshow(self.config.window_name_crowd, display_frame_crowd)
            elif self.previous_second_window_state and not current_second_window_state:
                # Hide the window when it was previously shown but now disabled
                cv2.destroyWindow(self.config.window_name_crowd)
            
            self.previous_second_window_state = current_second_window_state

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
