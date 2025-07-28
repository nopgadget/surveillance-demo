import os
import torch
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from ultralytics import YOLO

class ModelManager:
    def __init__(self, config):
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