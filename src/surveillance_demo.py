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
from .effect_processor import ASCIIEffect, FaceBlackoutEffect
from .gesture_recognizer import GestureRecognizer
from .video_source import VideoSource
from .modules import FrameProcessor, DrawingManager, UIRenderer, EffectsManager, InputHandler
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
        
        # Initialize modular components
        self.frame_processor = FrameProcessor(config, self.ui_manager, self.model_manager, self.thread_manager, self.stream_source)
        self.drawing_manager = DrawingManager(self.ui_manager, self.gesture_recognizer)
        self.ui_renderer = UIRenderer(self.ui_manager, self.gesture_recognizer)
        self.effects_manager = EffectsManager(self.ui_manager)
        self.input_handler = InputHandler(config, self.ui_manager)
        
        # Shared state
        self.current_finger_count = None
    
    def run(self):
        """Main application loop - much cleaner now!"""
        self.frame_processor.start_processing_threads()
        
        try:
            self._main_loop()
        finally:
            self._cleanup()
    
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
            display_frame = self.frame_processor.get_current_frame()
            
            if display_frame is None:
                time.sleep(0.01)
                continue
            
            # Get latest results from queues
            last_yolo_results = self.frame_processor.get_latest_results(self.frame_processor.yolo_results_queue, last_yolo_results)
            last_hand_results = self.frame_processor.get_latest_results(self.frame_processor.hand_results_queue, last_hand_results)
            last_face_mesh_results = self.frame_processor.get_latest_results(self.frame_processor.face_results_queue, last_face_mesh_results)
            last_pose_results = self.frame_processor.get_latest_results(self.frame_processor.pose_results_queue, last_pose_results)
            
            # Process gestures
            current_gesture, current_finger_count, thumbs_down_detected, middle_finger_detected = self.gesture_recognizer.process_gestures(
                last_hand_results, current_gesture, current_finger_count, thumbs_down_detected
            )
            
            # Store current finger count for menu highlighting
            self.current_finger_count = current_finger_count
            
            # Apply effects
            display_frame = self.effects_manager.apply_effects(display_frame, last_face_mesh_results)
            
            # Draw detections
            self.drawing_manager.draw_detections(display_frame, last_yolo_results, last_hand_results, last_pose_results, last_face_mesh_results)
            
            # Create clean frame for secondary window (with effects and detections, but no UI)
            crowd_frame = display_frame.copy()
            
            # Draw UI elements on the main display frame
            self.ui_renderer.draw_ui_elements(display_frame, self.current_finger_count)
            
            # Draw haptic text (on top of everything)
            self.ui_renderer.draw_haptic_text(display_frame)
            
            # Handle second window with clean frame
            self.input_handler.handle_second_window(crowd_frame)
            
            # Draw gesture progress
            self.ui_renderer.draw_gesture_progress(display_frame, current_gesture)
            
            # Draw finger count debug
            self.ui_renderer.draw_finger_count_debug(display_frame, current_finger_count)
            
            # Draw pause indicator
            self.ui_renderer.draw_pause_indicator(display_frame)
            
            # Show frame
            cv2.imshow(self.config.window_name_interactive, display_frame)
            
            # Handle input
            if not self.input_handler.handle_input():
                break
    
    def _cleanup(self):
        """Clean up resources."""
        print("Cleaning up resources...")
        self.thread_manager.stop_all()
        self.stream_source.release()
        self.effects_manager.cleanup()
        cv2.destroyAllWindows()
        print("Application closed.") 