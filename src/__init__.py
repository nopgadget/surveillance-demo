# Surveillance Demo Package
# This package contains all the classes for the surveillance demo application

from .app_config import StreamSource
from .app_config import AppConfig
from .video_source import VideoSource
from .model_manager import ModelManager
from .ui_manager import UIManager
from .effect_processor import EffectProcessor, ASCIIEffect, FaceBlackoutEffect, OptimizedFaceSwapEffect
from .gesture_recognizer import GestureRecognizer
from .thread_manager import ThreadManager
from .surveillance_demo import SurveillanceDemo
from .main import main

__all__ = [
    'StreamSource',
    'AppConfig', 
    'VideoSource',
    'ModelManager',
    'UIManager',
    'EffectProcessor',
    'ASCIIEffect',
    'FaceBlackoutEffect',
    'OptimizedFaceSwapEffect',
    'GestureRecognizer',
    'ThreadManager',
    'SurveillanceDemo',
    'main'
] 