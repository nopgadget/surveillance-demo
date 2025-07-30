import cv2
import numpy as np
import time
import screeninfo
from .stream_source import StreamSource

class UIManager:
    def __init__(self, config):
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
        # Determine if hand detection should be enabled by default
        # Disable hand detection for video files since gestures don't make sense
        hand_detection_enabled = self.config.stream_source != StreamSource.VIDEO.value
        
        checkboxes = {
            'hand_detection': {'checked': hand_detection_enabled, 'rect': (0, 0, 20, 20), 'label': 'Hand Detection'},
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
    
 