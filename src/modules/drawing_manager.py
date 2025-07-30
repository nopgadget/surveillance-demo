import cv2
import numpy as np
import time
import cvzone
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

class DrawingManager:
    """Handles all drawing and rendering functionality."""
    
    def __init__(self, ui_manager, gesture_recognizer):
        self.ui_manager = ui_manager
        self.gesture_recognizer = gesture_recognizer
    
    def draw_detections(self, frame, yolo_results, hand_results, pose_results, face_mesh_results):
        """Draw all detections on the frame."""
        # Draw YOLO detections
        if yolo_results:
            boxes = yolo_results.boxes.xyxy.int().cpu().tolist()
            track_ids = yolo_results.boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cvzone.putTextRect(frame, f"{track_id}", (max(0, x1 + 10), max(35, y1 - 10)), 
                                 scale=0.3, thickness=0, colorT=(0, 0, 0), colorR=(255, 255, 255))
        
        # Draw hand landmarks
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
        
        # Draw pose landmarks
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
        
        # Draw face mesh landmarks
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