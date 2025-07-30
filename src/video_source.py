import cv2
from .app_config import StreamSource

class VideoSource:
    def __init__(self, config):
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