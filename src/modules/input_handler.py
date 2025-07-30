import cv2
import time

class InputHandler:
    """Handles all input processing and window management."""
    
    def __init__(self, config, ui_manager):
        self.config = config
        self.ui_manager = ui_manager
        self.previous_second_window_state = False
    
    def handle_input(self):
        """Handle keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or escape key
            return False
        elif key == ord(' '):  # Space bar - toggle pause
            self.ui_manager.toggle_video_pause()
        return True
    
    def handle_second_window(self, crowd_frame):
        """Handle second window visibility."""
        current_second_window_state = self.ui_manager.checkboxes['second_window']['checked']
        
        if current_second_window_state:
            # Create the window if it doesn't exist
            if not self.previous_second_window_state:
                if self.config.fullscreen:
                    cv2.namedWindow(self.config.window_name_crowd, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(self.config.window_name_crowd, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.namedWindow(self.config.window_name_crowd, cv2.WINDOW_AUTOSIZE)
                    cv2.resizeWindow(self.config.window_name_crowd, self.ui_manager.width, self.ui_manager.height)
            
            # Display the clean frame (with effects and detections, but no UI elements)
            cv2.imshow(self.config.window_name_crowd, crowd_frame)
        elif self.previous_second_window_state and not current_second_window_state:
            # Hide the window when it was previously shown but now disabled
            cv2.destroyWindow(self.config.window_name_crowd)
        
        self.previous_second_window_state = current_second_window_state 