import cv2
import numpy as np
import time
import threading
import queue
from queue import Empty
import cvzone
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
from ultralytics import YOLO

from .app_config import AppConfig
from .ui_manager import UIManager
from .model_manager import ModelManager
from .thread_manager import ThreadManager
from .effect_processor import ASCIIEffect, FaceOverlayEffect, FaceBlackoutEffect
from .gesture_recognizer import GestureRecognizer
from .video_source import VideoSource
from pathlib import Path

class SurveillanceDemo:
    def __init__(self, config):
        self.config = config
        # Initialize components
        self.ui_manager = UIManager(config)
        self.model_manager = ModelManager(config)
        self.thread_manager = ThreadManager()
        self.gesture_recognizer = GestureRecognizer(self.ui_manager)
        self.stream_source = VideoSource(config)
        
        # Initialize effects
        self.effects = {
            'ascii': ASCIIEffect(),
            'face_overlay': FaceOverlayEffect(self.ui_manager.assets['face_overlay']),
            'face_blackout': FaceBlackoutEffect()
        }
        
        # Shared state
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Queues for results
        self.yolo_results_queue = queue.Queue(maxsize=1)
        self.hand_results_queue = queue.Queue(maxsize=1)
        self.face_results_queue = queue.Queue(maxsize=1)
        self.pose_results_queue = queue.Queue(maxsize=1)
    
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
            
            ret, frame = self.stream_source.read()
            if not ret:
                print("End of stream or camera disconnected.")
                self.thread_manager.stop_all()
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
            current_gesture, current_finger_count, thumbs_down_detected, middle_finger_detected = self.gesture_recognizer.process_gestures(
                last_hand_results, current_gesture, current_finger_count, thumbs_down_detected
            )
            
            # Apply effects
            display_frame = self._apply_effects(display_frame, last_face_mesh_results)
            
            # Draw detections
            self._draw_detections(display_frame, last_yolo_results, last_hand_results, last_pose_results, last_face_mesh_results)
            
            # Draw UI elements
            self._draw_ui_elements(display_frame)
            
            # Draw haptic text (on top of everything)
            self._draw_haptic_text(display_frame)
            
            # Handle second window
            self._handle_second_window(display_frame)
            
            # Draw gesture progress
            self._draw_gesture_progress(display_frame, current_gesture)
            
            # Draw finger count debug
            self._draw_finger_count_debug(display_frame, current_finger_count)
            
            # Draw pause indicator
            self._draw_pause_indicator(display_frame)
            
            # Show frame
            cv2.imshow(self.config.window_name_interactive, display_frame)
            
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
        
        # Apply face blackout
        if self.ui_manager.checkboxes['face_blackout']['checked']:
            frame = self.effects['face_blackout'].process(frame, face_mesh_results=face_mesh_results)
        
        return frame
    
    def _draw_detections(self, frame, yolo_results, hand_results, pose_results, face_mesh_results):
        # Draw YOLO detections
        if yolo_results:
            boxes = yolo_results.boxes.xyxy.int().cpu().tolist()
            track_ids = yolo_results.boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cvzone.putTextRect(frame, f"{track_id}", (max(0, x1 + 10), max(35, y1 - 10)), 
                                 scale=0.3, thickness=0, colorT=(0, 0, 0), colorR=(255, 255, 255))
        
        # Draw hand landmarks (restored from original test1.py)
        orange_spec = mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=2, circle_radius=4)  # BGR: orange
        white_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4)
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if self.gesture_recognizer._is_middle_finger(hand_landmarks):
                    # Draw black box with "CENSORED" text over the hand
                    self._draw_censored_box(frame, hand_landmarks)
                elif self.gesture_recognizer._is_thumbs_down(hand_landmarks):
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                        orange_spec, orange_spec
                    )
                else:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                        white_spec, white_spec
                    )
        
        # Draw pose landmarks (restored from original test1.py)
        if pose_results and pose_results.pose_landmarks and self.ui_manager.checkboxes['pose_detection']['checked']:
            # Draw up to the wrists (shoulders, elbows, wrists, torso, legs), exclude hand/finger landmarks (17-22)
            pose_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            index_map = {orig_idx: i for i, orig_idx in enumerate(pose_indices)}
            # Filter connections to only those with both indices in pose_indices
            pose_connections = [
                (index_map[a], index_map[b])
                for (a, b) in list(mp_pose.POSE_CONNECTIONS)
                if a in pose_indices and b in pose_indices
            ]
            from mediapipe.framework.formats import landmark_pb2
            NormalizedLandmarkList = getattr(landmark_pb2, 'NormalizedLandmarkList')
            all_landmarks = pose_results.pose_landmarks.landmark
            pose_landmarks = NormalizedLandmarkList(
                landmark=[all_landmarks[i] for i in pose_indices]
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=pose_landmarks,
                connections=pose_connections,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
            )
        
        # Draw face mesh landmarks (restored from original test1.py)
        if self.ui_manager.checkboxes['face_mesh']['checked'] and face_mesh_results:
            if (hasattr(face_mesh_results, 'multi_face_landmarks') and
                face_mesh_results.multi_face_landmarks is not None and
                isinstance(face_mesh_results.multi_face_landmarks, (list, tuple))):
                
                def filter_connections(connections, num_landmarks):
                    return [conn for conn in connections if max(conn) < num_landmarks]
                
                for face_landmarks in list(face_mesh_results.multi_face_landmarks):
                    num_landmarks = len(face_landmarks.landmark)
                    tess = filter_connections(mp.solutions.face_mesh.FACEMESH_TESSELATION, num_landmarks)
                    contours = filter_connections(mp.solutions.face_mesh.FACEMESH_CONTOURS, num_landmarks)
                    irises = filter_connections(mp.solutions.face_mesh.FACEMESH_IRISES, num_landmarks)
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=tess,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=0),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(150,150,150), thickness=1, circle_radius=1)
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=contours,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=1)
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=irises,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=1)
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
        """Draws haptic text messages on the left side of the screen."""
        if self.gesture_recognizer.haptic_text is None or self.gesture_recognizer.haptic_text_start_time is None:
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.gesture_recognizer.haptic_text_start_time
        
        if elapsed_time > self.gesture_recognizer.haptic_text_duration:
            # Clear the haptic text
            self.gesture_recognizer.haptic_text = None
            self.gesture_recognizer.haptic_text_start_time = None
            self.gesture_recognizer.haptic_text_alpha = 0.0
            return
            
        # Calculate alpha for fade in/out effect
        if elapsed_time < 0.5:
            # Fade in
            self.gesture_recognizer.haptic_text_alpha = elapsed_time / 0.5
        elif elapsed_time > self.gesture_recognizer.haptic_text_duration - 0.5:
            # Fade out
            self.gesture_recognizer.haptic_text_alpha = (self.gesture_recognizer.haptic_text_duration - elapsed_time) / 0.5
        else:
            # Full opacity
            self.gesture_recognizer.haptic_text_alpha = 1.0
            
        # Clamp alpha to 0-1
        self.gesture_recognizer.haptic_text_alpha = max(0.0, min(1.0, self.gesture_recognizer.haptic_text_alpha))
        
        # Draw the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.4  # Reduced from 1.2 to 0.24 (1/5 of original size)
        font_thickness = 1  # Reduced from 3 to 1 for smaller text
        
        # Calculate text size and position (left side, vertically centered)
        text_size, _ = cv2.getTextSize(self.gesture_recognizer.haptic_text, font, text_scale, font_thickness)
        text_x = 50  # Left margin
        text_y = (frame.shape[0] + text_size[1]) // 2  # Vertically centered
        
        # Draw background rectangle for better visibility
        bg_padding = 20
        bg_alpha = self.gesture_recognizer.haptic_text_alpha * 0.7  # Slightly transparent background
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
        text_color_with_alpha = tuple(int(c * self.gesture_recognizer.haptic_text_alpha) for c in text_color)
        
        cv2.putText(frame, self.gesture_recognizer.haptic_text, (text_x, text_y), font, text_scale, 
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
            
            # Create crowd frame (copy of interactive frame without UI elements)
            crowd_frame = display_frame.copy()
            
            # Note: No UI elements (info text, QR code, etc.) are drawn on the secondary window
            # Only the raw video feed with detections and effects is shown
            
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
        self.stream_source.release()
        cv2.destroyAllWindows()
        print("Application closed.") 

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