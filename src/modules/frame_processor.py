import cv2
import numpy as np
import time
import threading
import queue
from queue import Empty
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
from ultralytics import YOLO

class FrameProcessor:
    """Handles all frame processing threads and related functionality."""
    
    def __init__(self, config, ui_manager, model_manager, thread_manager, stream_source):
        self.config = config
        self.ui_manager = ui_manager
        self.model_manager = model_manager
        self.thread_manager = thread_manager
        self.stream_source = stream_source
        
        # Shared state
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Queues for results
        self.yolo_results_queue = queue.Queue(maxsize=1)
        self.hand_results_queue = queue.Queue(maxsize=1)
        self.face_results_queue = queue.Queue(maxsize=1)
        self.pose_results_queue = queue.Queue(maxsize=1)
    
    def start_processing_threads(self):
        """Start all processing threads."""
        self.thread_manager.add_thread(self._frame_reader_thread)
        self.thread_manager.add_thread(self._yolo_processor_thread)
        self.thread_manager.add_thread(self._hand_processor_thread)
        self.thread_manager.add_thread(self._face_mesh_processor_thread)
        self.thread_manager.add_thread(self._pose_processor_thread)
        self.thread_manager.start_all()
    
    def _frame_reader_thread(self):
        """Reads frames from video source."""
        frame_count = 0
        start_time = time.time()
        
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
            
            # Update FPS based on actual frame reading
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time >= 0.5:  # Update every 0.5 seconds
                fps = frame_count / elapsed_time
                self.ui_manager.fps_string = f"FPS: {int(fps)}"
                frame_count = 0
                start_time = current_time
            
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
    
    def get_latest_results(self, queue_obj, last_results):
        """Get the latest results from a queue."""
        try:
            return queue_obj.get_nowait()
        except queue.Empty:
            return last_results
    
    def get_current_frame(self):
        """Get the current frame with thread safety."""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None 