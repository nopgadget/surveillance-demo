import cv2
import numpy as np
import time

class UIRenderer:
    """Handles all UI rendering functionality."""
    
    def __init__(self, ui_manager, gesture_recognizer):
        self.ui_manager = ui_manager
        self.gesture_recognizer = gesture_recognizer
    
    def draw_ui_elements(self, frame, current_finger_count=None):
        """Draw all UI elements on the frame."""
        # Draw info text
        if self.ui_manager.checkboxes['info_display']['checked']:
            self._draw_info_text(frame, self.ui_manager.config.info_text_interactive)
        
        # Draw QR code
        self._overlay_image(frame, self.ui_manager.assets['qr_code'], position="bottom-left")
        
        # Draw finger count menu above QR code
        self.ui_manager.draw_finger_menu(frame, current_finger_count)
        
        # Draw checkboxes
        if self.ui_manager.checkboxes_visible:
            self._draw_checkboxes(frame)
        
        # Draw FPS counter
        if self.ui_manager.checkboxes['fps_counter']['checked']:
            self._draw_fps_counter(frame)
    
    def _draw_info_text(self, frame, text):
        """Draw info text with fade effect."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale, font_thickness = 1, 2
        text_size, _ = cv2.getTextSize(text, font, text_scale, font_thickness)
        
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = text_size[1] + 20
        
        fade = 0.5 * (1 + np.sin(time.time() * 2))
        color = (int(fade * 230), int(fade * 216), int(fade * 173))
        cv2.putText(frame, text, (text_x, text_y), font, text_scale, color, font_thickness, cv2.LINE_AA)
    
    def _overlay_image(self, frame, overlay_img, position="bottom-right", margin=10):
        """Overlay an image on the frame."""
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
        """Draw checkboxes on the frame."""
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
    
    def draw_gesture_progress(self, frame, gesture):
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
    
    def draw_finger_count_debug(self, frame, finger_count):
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
    
    def draw_haptic_text(self, frame):
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
    
    def draw_pause_indicator(self, frame):
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