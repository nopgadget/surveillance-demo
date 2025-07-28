import os
import pytomlpp
from dataclasses import dataclass

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