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
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from mediapipe.framework.formats import landmark_pb2

# --- Constants ---
class StreamSource(Enum):
    RTSP = "rtsp"
    WEBCAM = "webcam"
    VIDEO = "video"

# --- Configuration ---
@dataclass
class AppConfig:
    stream_source: str = "rtsp"
    rtsp_url: str = "rtsp://192.168.1.109:554/0/0/0"
    webcam_id: int = 0
    video_path: str = "vid/hongdae.mp4"
    fullscreen: bool = False
    window_name: str = "Multi-Tracking Demo"  # Added to match config
    window_name_crowd: str = "People Counter"
    window_name_interactive: str = "Interactive"
    info_text: str = "No data is retained, stored or shared."
    info_text_interactive: str = "No data is retained, stored or shared. Use hand gestures for effects."
    logo_path: str = "img/odplogo.png"
    qr_code_path: str = "img/qr-code.png"
    face_overlay_path: str = "img/dwayne.png"  # Updated to match config
    face_swap_source_path: str = "img/musk.jpg"  # Added to match config
    model_path: str = "models/yolo11n"
    yolo_conf_threshold: float = 0.3

    @classmethod
    def from_toml(cls, config_path: str = "config.toml") -> 'AppConfig':
        if not os.path.exists(config_path):
            config_path = "example-config.toml"
        
        with open(config_path) as config_file:
            config_toml = pytomlpp.load(config_file)
        
        if not isinstance(config_toml, dict):
            return cls()
        
        # Filter only the fields that exist in our dataclass
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_toml.items() if k in valid_fields}
        
        return cls(**filtered_config)

# --- Video Source Management ---
class VideoSource:
    def __init__(self, config: AppConfig):
        self.config = config
        self.cap = None
        self._setup_video_source()
    
    def _setup_video_source(self):
        source = self._get_source_from_config()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video stream: {source}")
    
    def _get_source_from_config(self):
        if self.config.stream_source == StreamSource.WEBCAM.value:
            return self.config.webcam_id
        elif self.config.stream_source == StreamSource.VIDEO.value:
            return self.config.video_path
        elif self.config.stream_source == StreamSource.RTSP.value:
            return self.config.rtsp_url
        else:
            raise RuntimeError(f"Unsupported video source: {self.config.stream_source}")
    
    def read(self):
        return self.cap.read()
    
    def release(self):
        if self.cap:
            self.cap.release()

# --- Model Management ---
class ModelManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.yolo_model = None
        self.hands_model = None
        self.face_mesh = None
        self.pose_model = None
        self._setup_models()
    
    def _setup_models(self):
        self._setup_yolo()
        self._setup_mediapipe_models()
    
    def _setup_yolo(self):
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
    
    def _setup_mediapipe_models(self):
        # Setup hands
        self.hands_model = mp.solutions.hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=10
        )
        
        # Setup face mesh
        try:
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print(f"Warning: Could not import mediapipe face_mesh: {e}")
            self.face_mesh = None
        
        # Setup pose
        self.pose_model = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

# --- UI Management ---
class UIManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.width, self.height = self._setup_screen()
        self.assets = self._load_assets()
        self.checkboxes = self._setup_checkboxes()
        self.checkboxes_visible = False
        
        # Video pause state
        self.video_paused = False
        self.is_video_file = config.stream_source == StreamSource.VIDEO.value
        
        # Haptic text system
        self.haptic_text = None
        self.haptic_text_start_time = None
        self.haptic_text_duration = 3.0
        self.haptic_text_alpha = 0.0
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_string = "FPS: N/A"
        
        # Set mouse callback
        cv2.setMouseCallback(self.config.window_name_interactive, self._mouse_callback)
    
    def _setup_screen(self):
        if self.config.fullscreen:
            try:
                screen = screeninfo.get_monitors()[0]
                width, height = screen.width, screen.height
                cv2.namedWindow(self.config.window_name_interactive, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(self.config.window_name_interactive, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except screeninfo.common.ScreenInfoError:
                print("Could not get screen info. Using 1280x720.")
                width, height = 1280, 720
                cv2.namedWindow(self.config.window_name_interactive, cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow(self.config.window_name_interactive, width, height)
        else:
            width, height = 1280, 720
            cv2.namedWindow(self.config.window_name_interactive, cv2.WINDOW_AUTOSIZE)
            cv2.resizeWindow(self.config.window_name_interactive, width, height)
        
        return width, height
    
    def _load_assets(self):
        return {
            'logo': self._load_image(self.config.logo_path),
            'qr_code': self._load_image(self.config.qr_code_path),
            'face_overlay': self._load_image(self.config.face_overlay_path)
        }
    
    def _load_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Image not found at {path}. Creating a placeholder.")
            return np.zeros((100, 100, 4), dtype=np.uint8)
        return img
    
    def _setup_checkboxes(self):
        checkboxes = {
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
        self._update_checkbox_positions(checkboxes)
        return checkboxes
    
    def _update_checkbox_positions(self, checkboxes):
        checkbox_width = 20
        checkbox_height = 20
        checkbox_spacing = 40
        
        right_margin = 50
        checkbox_x = self.width - right_margin - checkbox_width - 150
        
        total_checkboxes = len(checkboxes)
        total_height = (total_checkboxes - 1) * checkbox_spacing + checkbox_height
        start_y = (self.height - total_height) // 2
        
        for i, (feature_name, checkbox) in enumerate(checkboxes.items()):
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
            qr_oh, qr_ow = self.assets['qr_code'].shape[:2] if self.assets['qr_code'] is not None else (100, 100)
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
    
    def toggle_video_pause(self):
        """Toggle video pause state (only for video files)."""
        if self.is_video_file:
            self.video_paused = not self.video_paused
    
    def show_haptic_text(self, message):
        """Shows a haptic text message."""
        self.haptic_text = message
        self.haptic_text_start_time = time.time()
        self.haptic_text_alpha = 0.0
    
    def update_fps(self):
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time >= 0.5:  # Update every 0.5 seconds
            fps = self.frame_count / elapsed_time
            self.fps_string = f"FPS: {int(fps)}"
            self.frame_count = 0
            self.start_time = current_time

# --- Effect Processors ---
class EffectProcessor(ABC):
    @abstractmethod
    def process(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        pass

class ASCIIEffect(EffectProcessor):
    def __init__(self):
        self.font_scale = 0.4
        self.cell_size = 8
        self.chars = "@%#*+=-:. "
    
    def process(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        out_img = np.zeros_like(frame)
        
        for y in range(0, h, self.cell_size):
            for x in range(0, w, self.cell_size):
                cell = gray[y:y+self.cell_size, x:x+self.cell_size]
                if cell.size == 0:
                    continue
                avg = int(np.mean(cell))
                char_idx = int((avg / 255) * (len(self.chars) - 1))
                char = self.chars[char_idx]
                
                green_intensity = int((avg / 255) * 255) + 100
                color = (0, green_intensity, 0)
                
                cv2.putText(
                    out_img, char, (x, y + self.cell_size),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 1, cv2.LINE_AA
                )
        return out_img

class FaceOverlayEffect(EffectProcessor):
    def __init__(self, face_overlay_img: np.ndarray):
        self.face_overlay = face_overlay_img
    
    def process(self, frame: np.ndarray, face_mesh_results=None, **kwargs) -> np.ndarray:
        if face_mesh_results is None or not face_mesh_results.multi_face_landmarks:
            return frame
        
        if self.face_overlay is None:
            return frame
        
        # Implementation would go here - simplified for brevity
        return frame

# --- Gesture Recognition ---
class GestureRecognizer:
    def __init__(self):
        self.gesture_start_time = None
        self.current_gesture = None
        self.gesture_duration = 2.0
        self.gesture_cooldown = 3.0
        self.last_gesture_time = 0
    
    def count_fingers(self, hand_landmarks) -> int:
        finger_tips = [
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
            mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
            mp.solutions.hands.HandLandmark.PINKY_TIP
        ]
        
        finger_pips = [
            mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
            mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
            mp.solutions.hands.HandLandmark.PINKY_PIP
        ]
        
        extended_fingers = 0
        for i in range(len(finger_tips)):
            tip_y = hand_landmarks.landmark[finger_tips[i]].y
            pip_y = hand_landmarks.landmark[finger_pips[i]].y
            if tip_y < pip_y - 0.01:
                extended_fingers += 1
        
        return extended_fingers
    
    def is_thumbs_down(self, hand_landmarks) -> bool:
        """Checks for a proper thumbs-down gesture."""
        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP]
        wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
        
        # Check that thumb is extended downward (tip below IP joint)
        if thumb_tip.y > thumb_ip.y:
            # Check that other fingers are curled
            finger_tips = [
                mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                mp.solutions.hands.HandLandmark.PINKY_TIP
            ]
            finger_pips = [
                mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
                mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
                mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
                mp.solutions.hands.HandLandmark.PINKY_PIP
            ]
            
            fingers_curled = True
            for i in range(len(finger_tips)):
                tip_y = hand_landmarks.landmark[finger_tips[i]].y
                pip_y = hand_landmarks.landmark[finger_pips[i]].y
                if tip_y < pip_y - 0.01:
                    fingers_curled = False
                    break
            
            # Check hand orientation and thumb position
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
            
            hand_width = abs(index_tip.x - pinky_tip.x)
            hand_height = abs(thumb_tip.y - wrist.y)
            
            # Check if thumb is lower than all other finger tips
            all_finger_tips = [
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP],
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP],
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP],
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
            ]
            
            thumb_is_lowest = True
            for finger_tip in all_finger_tips:
                if thumb_tip.y <= finger_tip.y:
                    thumb_is_lowest = False
                    break
            
            if (hand_height > hand_width * 0.8 and 
                thumb_tip.y > wrist.y + 0.05 and 
                thumb_is_lowest and 
                fingers_curled):
                return True
        
        return False
    
    def handle_gesture_activation(self, finger_count, ui_manager):
        """Handle gesture-based feature activation."""
        current_time = time.time()
        
        # Check if we're in cooldown period
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return
        
        # Map finger count to features
        gesture_features = {
            1: 'pose_detection',
            2: 'ascii_effect',
            3: 'face_mesh',
            4: 'face_blackout',
            5: 'face_overlay'
        }
        
        if finger_count in gesture_features:
            feature_name = gesture_features[finger_count]
            
            if feature_name in ui_manager.checkboxes and not ui_manager.checkboxes[feature_name]['checked']:
                ui_manager.checkboxes[feature_name]['checked'] = True
                ui_manager.show_haptic_text(f"{ui_manager.checkboxes[feature_name]['label']} enabled!")
                self.last_gesture_time = current_time
                print(f"Gesture {finger_count} fingers: {ui_manager.checkboxes[feature_name]['label']} enabled")
    
    def handle_gesture_deactivation(self, ui_manager):
        """Handle thumbs-down gesture to deactivate features."""
        current_time = time.time()
        
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return
        
        deactivatable_features = ['pose_detection', 'ascii_effect', 'face_mesh', 'face_blackout', 'face_overlay']
        
        deactivated_count = 0
        for feature_name in deactivatable_features:
            if feature_name in ui_manager.checkboxes and ui_manager.checkboxes[feature_name]['checked']:
                ui_manager.checkboxes[feature_name]['checked'] = False
                deactivated_count += 1
        
        if deactivated_count > 0:
            ui_manager.show_haptic_text(f"Deactivated {deactivated_count} features!")
            self.last_gesture_time = current_time
            print(f"Thumbs down: Deactivated {deactivated_count} features")

# --- Threading Management ---
class ThreadManager:
    def __init__(self):
        self.stop_event = threading.Event()
        self.threads = []
    
    def add_thread(self, target, daemon=True):
        thread = threading.Thread(target=target)
        thread.daemon = daemon
        self.threads.append(thread)
        return thread
    
    def start_all(self):
        for thread in self.threads:
            thread.start()
    
    def stop_all(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=2)

# --- Main Application ---
class SurveillanceDemo:
    def __init__(self, config: AppConfig):
        self.config = config
        self.thread_manager = ThreadManager()
        self.video_source = VideoSource(config)
        self.model_manager = ModelManager(config)
        self.ui_manager = UIManager(config)
        self.gesture_recognizer = GestureRecognizer()
        
        # Shared state
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Queues for results
        self.yolo_results_queue = queue.Queue(maxsize=1)
        self.hand_results_queue = queue.Queue(maxsize=1)
        self.face_results_queue = queue.Queue(maxsize=1)
        self.pose_results_queue = queue.Queue(maxsize=1)
        
        # Effect processors
        self.effects = {
            'ascii': ASCIIEffect(),
            'face_overlay': FaceOverlayEffect(self.ui_manager.assets['face_overlay'])
        }
    
    def run(self):
        """Main application loop - much cleaner now!"""
        self._start_processing_threads()
        
        try:
            self._main_loop()
        finally:
            self._cleanup()
    
    def _start_processing_threads(self):
        self.thread_manager.add_thread(self._frame_reader_thread)
        self.thread_manager.add_thread(self._yolo_processor_thread)
        self.thread_manager.add_thread(self._hand_processor_thread)
        self.thread_manager.add_thread(self._face_mesh_processor_thread)
        self.thread_manager.add_thread(self._pose_processor_thread)
        self.thread_manager.start_all()
    
    def _frame_reader_thread(self):
        """Reads frames from video source."""
        while not self.thread_manager.stop_event.is_set():
            # Check if video is paused (only for video files)
            if self.ui_manager.is_video_file and self.ui_manager.video_paused:
                time.sleep(0.01)  # Small delay when paused
                continue
            
            ret, frame = self.video_source.read()
            if not ret:
                print("End of stream or camera disconnected.")
                self.thread_manager.stop_event.set()
                break
            
            resized_frame = cv2.resize(frame, (self.ui_manager.width, self.ui_manager.height))
            with self.frame_lock:
                self.latest_frame = resized_frame.copy()
    
    def _yolo_processor_thread(self):
        """Processes frames with YOLO model."""
        while not self.thread_manager.stop_event.is_set():
            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            
            if frame_to_process is not None:
                results = self.model_manager.yolo_model.track(
                    frame_to_process, 
                    persist=True, 
                    classes=0, 
                    verbose=False, 
                    conf=self.config.yolo_conf_threshold
                )
                if results and results[0].boxes and results[0].boxes.id is not None:
                    if not self.yolo_results_queue.empty():
                        try:
                            self.yolo_results_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.yolo_results_queue.put(results[0])
            
            time.sleep(0.02)
    
    def _hand_processor_thread(self):
        """Processes frames with MediaPipe hands."""
        while not self.thread_manager.stop_event.is_set():
            if not self.ui_manager.checkboxes['hand_detection']['checked']:
                time.sleep(0.1)
                continue
            
            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            
            if frame_to_process is not None:
                img_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = self.model_manager.hands_model.process(img_rgb)
                
                if not self.hand_results_queue.empty():
                    try:
                        self.hand_results_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.hand_results_queue.put(results)
            
            time.sleep(0.02)
    
    def _face_mesh_processor_thread(self):
        """Processes frames with MediaPipe face mesh."""
        if self.model_manager.face_mesh is None:
            return
        
        while not self.thread_manager.stop_event.is_set():
            if not (self.ui_manager.checkboxes['face_mesh']['checked'] or 
                   self.ui_manager.checkboxes['face_overlay']['checked'] or 
                   self.ui_manager.checkboxes['face_blackout']['checked']):
                time.sleep(0.1)
                continue
            
            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            
            if frame_to_process is not None:
                img_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = self.model_manager.face_mesh.process(img_rgb)
                
                if not self.face_results_queue.empty():
                    try:
                        self.face_results_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.face_results_queue.put(results)
            
            time.sleep(0.02)
    
    def _pose_processor_thread(self):
        """Processes frames with MediaPipe pose."""
        while not self.thread_manager.stop_event.is_set():
            if not self.ui_manager.checkboxes['pose_detection']['checked']:
                time.sleep(0.1)
                continue
            
            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            
            if frame_to_process is not None:
                img_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = self.model_manager.pose_model.process(img_rgb)
                
                if not self.pose_results_queue.empty():
                    try:
                        self.pose_results_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.pose_results_queue.put(results)
            
            time.sleep(0.02)
    
    def _main_loop(self):
        """Main rendering and interaction loop."""
        last_yolo_results = None
        last_hand_results = None
        last_face_mesh_results = None
        last_pose_results = None
        
        # Gesture tracking variables
        current_gesture = None
        current_finger_count = None
        thumbs_down_detected = False
        
        while not self.thread_manager.stop_event.is_set():
            # Get current frame
            display_frame = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    display_frame = self.latest_frame.copy()
            
            if display_frame is None:
                time.sleep(0.01)
                continue
            
            # Get latest results from queues
            last_yolo_results = self._get_latest_results(self.yolo_results_queue, last_yolo_results)
            last_hand_results = self._get_latest_results(self.hand_results_queue, last_hand_results)
            last_face_mesh_results = self._get_latest_results(self.face_results_queue, last_face_mesh_results)
            last_pose_results = self._get_latest_results(self.pose_results_queue, last_pose_results)
            
            # Process gestures
            current_gesture, current_finger_count, thumbs_down_detected = self._process_gestures(
                last_hand_results, current_gesture, current_finger_count, thumbs_down_detected
            )
            
            # Apply effects
            display_frame = self._apply_effects(display_frame, last_face_mesh_results)
            
            # Draw detections
            self._draw_detections(display_frame, last_yolo_results, last_hand_results, last_pose_results)
            
            # Draw UI elements
            self._draw_ui_elements(display_frame)
            
            # Draw gesture progress
            self._draw_gesture_progress(display_frame, current_gesture)
            
            # Draw finger count debug
            self._draw_finger_count_debug(display_frame, current_finger_count)
            
            # Draw haptic text
            self._draw_haptic_text(display_frame)
            
            # Draw pause indicator
            self._draw_pause_indicator(display_frame)
            
            # Show frame
            cv2.imshow(self.config.window_name_interactive, display_frame)
            
            # Handle second window
            self._handle_second_window(display_frame)
            
            # Handle input
            if not self._handle_input():
                break
    
    def _get_latest_results(self, queue_obj, last_results):
        try:
            return queue_obj.get_nowait()
        except queue.Empty:
            return last_results
    
    def _apply_effects(self, frame, face_mesh_results):
        # Apply ASCII effect
        if self.ui_manager.checkboxes['ascii_effect']['checked']:
            frame = self.effects['ascii'].process(frame)
        
        # Apply face overlay
        if self.ui_manager.checkboxes['face_overlay']['checked']:
            frame = self.effects['face_overlay'].process(frame, face_mesh_results=face_mesh_results)
        
        return frame
    
    def _draw_detections(self, frame, yolo_results, hand_results, pose_results):
        # Draw YOLO detections
        if yolo_results:
            boxes = yolo_results.boxes.xyxy.int().cpu().tolist()
            track_ids = yolo_results.boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cvzone.putTextRect(frame, f"{track_id}", (max(0, x1 + 10), max(35, y1 - 10)), 
                                 scale=0.3, thickness=0, colorT=(0, 0, 0), colorR=(255, 255, 255))
        
        # Draw hand landmarks (simplified)
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4)
                )
    
    def _draw_ui_elements(self, frame):
        # Update FPS counter
        self.ui_manager.update_fps()
        
        # Draw info text
        if self.ui_manager.checkboxes['info_display']['checked']:
            self._draw_info_text(frame, self.config.info_text_interactive)
        
        # Draw QR code
        self._overlay_image(frame, self.ui_manager.assets['qr_code'], position="bottom-left")
        
        # Draw checkboxes
        if self.ui_manager.checkboxes_visible:
            self._draw_checkboxes(frame)
        
        # Draw FPS counter
        if self.ui_manager.checkboxes['fps_counter']['checked']:
            self._draw_fps_counter(frame)
    
    def _draw_info_text(self, frame, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale, font_thickness = 1, 2
        text_size, _ = cv2.getTextSize(text, font, text_scale, font_thickness)
        
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = text_size[1] + 20
        
        fade = 0.5 * (1 + np.sin(time.time() * 2))
        color = (int(fade * 230), int(fade * 216), int(fade * 173))
        cv2.putText(frame, text, (text_x, text_y), font, text_scale, color, font_thickness, cv2.LINE_AA)
    
    def _overlay_image(self, frame, overlay_img, position="bottom-right", margin=10):
        if overlay_img is None or overlay_img.shape[0] == 0 or overlay_img.shape[1] == 0:
            return
        
        fh, fw, _ = frame.shape
        oh, ow, _ = overlay_img.shape
        max_width = fw // 6
        if ow > max_width:
            scale = max_width / ow
            overlay_img = cv2.resize(overlay_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            oh, ow, _ = overlay_img.shape
        
        if position == "bottom-right":
            x, y = fw - ow - margin, fh - oh - margin
        elif position == "bottom-left":
            x, y = margin, fh - oh - margin
        
        roi = frame[y:y+oh, x:x+ow]
        if overlay_img.shape[2] == 4:
            alpha = overlay_img[:, :, 3] / 255.0
            for c in range(0, 3):
                roi[:, :, c] = overlay_img[:, :, c] * alpha + roi[:, :, c] * (1.0 - alpha)
        else:
            roi[:] = overlay_img
        
        frame[y:y+oh, x:x+ow] = roi
    
    def _draw_checkboxes(self, frame):
        for feature_name, checkbox in self.ui_manager.checkboxes.items():
            x, y, w, h = checkbox['rect']
            
            # Draw checkbox border
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            # Draw checkmark if checked
            if checkbox['checked']:
                cv2.line(frame, (x + 3, y + h // 2), (x + w // 3, y + h - 3), (0, 255, 0), 2)
                cv2.line(frame, (x + w // 3, y + h - 3), (x + w - 3, y + 3), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(frame, checkbox['label'], (x + w + 10, y + h // 2 + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_fps_counter(self, frame):
        """Draw FPS counter."""
        fps_x = self.ui_manager.width - self.ui_manager.width // 10
        fps_y = self.ui_manager.height - 40
        
        # Draw background rectangle
        fps_w, fps_h = cv2.getTextSize(self.ui_manager.fps_string, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (fps_x - 5, fps_y + 5), (fps_x + fps_w + 5, fps_y - fps_h - 5), (0, 0, 0), -1)
        
        # Draw FPS text
        cv2.putText(frame, self.ui_manager.fps_string, (fps_x, fps_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    def _process_gestures(self, hand_results, current_gesture, current_finger_count, thumbs_down_detected):
        """Process hand gestures and update gesture state."""
        if (self.ui_manager.checkboxes['hand_detection']['checked'] and 
            hand_results and hand_results.multi_hand_landmarks):
            
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Check for thumbs down gesture
                if self.gesture_recognizer.is_thumbs_down(hand_landmarks):
                    thumbs_down_detected = True
                    self.gesture_recognizer.handle_gesture_deactivation(self.ui_manager)
                    # Reset gesture tracking
                    current_gesture = None
                    self.gesture_recognizer.gesture_start_time = None
                    self.gesture_recognizer.current_gesture = None
                else:
                    # Count fingers for gesture detection
                    finger_count = self.gesture_recognizer.count_fingers(hand_landmarks)
                    current_finger_count = finger_count
                    if 1 <= finger_count <= 5:
                        current_gesture = finger_count
        
        # Handle gesture duration logic
        now = time.time()
        if current_gesture is not None and not thumbs_down_detected:
            gesture_features = {
                1: 'pose_detection',
                2: 'ascii_effect',
                3: 'face_mesh',
                4: 'face_blackout',
                5: 'face_overlay'
            }
            
            feature_name = gesture_features.get(current_gesture)
            feature_already_enabled = (feature_name in self.ui_manager.checkboxes and 
                                    self.ui_manager.checkboxes[feature_name]['checked'])
            
            if feature_already_enabled:
                # Reset gesture tracking for already enabled features
                self.gesture_recognizer.gesture_start_time = None
                self.gesture_recognizer.current_gesture = None
                current_gesture = None
            else:
                # Normal gesture tracking for disabled features
                if (self.gesture_recognizer.gesture_start_time is None or 
                    self.gesture_recognizer.current_gesture != current_gesture):
                    self.gesture_recognizer.gesture_start_time = now
                    self.gesture_recognizer.current_gesture = current_gesture
                elif now - self.gesture_recognizer.gesture_start_time >= self.gesture_recognizer.gesture_duration:
                    # Gesture held for required duration, trigger activation
                    self.gesture_recognizer.handle_gesture_activation(current_gesture, self.ui_manager)
                    # Reset gesture tracking
                    self.gesture_recognizer.gesture_start_time = None
                    self.gesture_recognizer.current_gesture = None
                    current_gesture = None
        elif thumbs_down_detected:
            # Reset gesture tracking after deactivation
            self.gesture_recognizer.gesture_start_time = None
            self.gesture_recognizer.current_gesture = None
        else:
            # No gesture detected, reset tracking
            self.gesture_recognizer.gesture_start_time = None
            self.gesture_recognizer.current_gesture = None
        
        return current_gesture, current_finger_count, thumbs_down_detected
    
    def _draw_gesture_progress(self, frame, gesture):
        """Draw gesture progress indicator."""
        if gesture is None or self.gesture_recognizer.gesture_start_time is None:
            return
        
        now = time.time()
        elapsed = now - self.gesture_recognizer.gesture_start_time
        progress = min(elapsed / self.gesture_recognizer.gesture_duration, 1.0)
        
        if progress <= 0:
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
    
    def _draw_haptic_text(self, frame):
        """Draws haptic text messages."""
        if self.ui_manager.haptic_text is None or self.ui_manager.haptic_text_start_time is None:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.ui_manager.haptic_text_start_time
        
        if elapsed_time > self.ui_manager.haptic_text_duration:
            # Clear the haptic text
            self.ui_manager.haptic_text = None
            self.ui_manager.haptic_text_start_time = None
            self.ui_manager.haptic_text_alpha = 0.0
            return
        
        # Calculate alpha for fade in/out effect
        if elapsed_time < 0.5:
            # Fade in
            self.ui_manager.haptic_text_alpha = elapsed_time / 0.5
        elif elapsed_time > self.ui_manager.haptic_text_duration - 0.5:
            # Fade out
            self.ui_manager.haptic_text_alpha = (self.ui_manager.haptic_text_duration - elapsed_time) / 0.5
        else:
            # Full opacity
            self.ui_manager.haptic_text_alpha = 1.0
        
        # Clamp alpha to 0-1
        self.ui_manager.haptic_text_alpha = max(0.0, min(1.0, self.ui_manager.haptic_text_alpha))
        
        # Draw the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.4
        font_thickness = 1
        
        # Calculate text size and position (left side, vertically centered)
        text_size, _ = cv2.getTextSize(self.ui_manager.haptic_text, font, text_scale, font_thickness)
        text_x = 50  # Left margin
        text_y = (frame.shape[0] + text_size[1]) // 2  # Vertically centered
        
        # Draw background rectangle for better visibility
        bg_padding = 20
        bg_alpha = self.ui_manager.haptic_text_alpha * 0.7
        bg_color = (int(0 * bg_alpha), int(0 * bg_alpha), int(0 * bg_alpha))
        
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
        text_color_with_alpha = tuple(int(c * self.ui_manager.haptic_text_alpha) for c in text_color)
        
        cv2.putText(frame, self.ui_manager.haptic_text, (text_x, text_y), font, text_scale, 
                   text_color_with_alpha, font_thickness, cv2.LINE_AA)
    
    def _draw_pause_indicator(self, frame):
        """Draw pause indicator for video files."""
        if self.ui_manager.is_video_file and self.ui_manager.video_paused:
            # Draw pause indicator in top-left corner
            pause_text = "PAUSED"
            text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x, text_y = 20, 40
            
            # Draw background rectangle
            cv2.rectangle(frame, 
                        (text_x - 10, text_y - text_size[1] - 10),
                        (text_x + text_size[0] + 10, text_y + 10),
                        (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, pause_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    def _handle_second_window(self, display_frame):
        """Handle second window visibility."""
        current_second_window_state = self.ui_manager.checkboxes['second_window']['checked']
        
        if current_second_window_state:
            # Create the window if it doesn't exist
            if not hasattr(self, 'previous_second_window_state') or not self.previous_second_window_state:
                if self.config.fullscreen:
                    cv2.namedWindow(self.config.window_name_crowd, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(self.config.window_name_crowd, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.namedWindow(self.config.window_name_crowd, cv2.WINDOW_AUTOSIZE)
                    cv2.resizeWindow(self.config.window_name_crowd, self.ui_manager.width, self.ui_manager.height)
            
            # Create crowd frame (copy of interactive frame)
            crowd_frame = display_frame.copy()
            
            # Draw info text and QR code on crowd window
            self._draw_info_text(crowd_frame, self.config.info_text)
            self._overlay_image(crowd_frame, self.ui_manager.assets['qr_code'], position="bottom-left")
            
            # Draw pause indicator on crowd window
            self._draw_pause_indicator(crowd_frame)
            
            cv2.imshow(self.config.window_name_crowd, crowd_frame)
        elif hasattr(self, 'previous_second_window_state') and self.previous_second_window_state and not current_second_window_state:
            # Hide the window when it was previously shown but now disabled
            cv2.destroyWindow(self.config.window_name_crowd)
        
        self.previous_second_window_state = current_second_window_state
    
    def _handle_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or escape key
            return False
        elif key == ord(' '):  # Space bar - toggle pause
            self.ui_manager.toggle_video_pause()
        return True
    
    def _cleanup(self):
        """Clean up resources."""
        print("Cleaning up resources...")
        self.thread_manager.stop_all()
        self.video_source.release()
        cv2.destroyAllWindows()
        print("Application closed.")

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    Path("img").mkdir(exist_ok=True)
    
    try:
        config = AppConfig.from_toml()
        app = SurveillanceDemo(config)
        app.run()
    except Exception as e:
        print(f"An error occurred: {e}") 