import cv2
import numpy as np
import time
import mediapipe as mp

class GestureRecognizer:
    def __init__(self, ui_manager):
        self.ui_manager = ui_manager
        self.gesture_start_time = None
        self.current_gesture = None
        self.gesture_duration = 2.0
        self.gesture_cooldown = 3.0
        self.last_gesture_time = 0
        
        # ASCII effect variables
        self.ascii_start_time = None
        self.ascii_duration = 10.0
        
        # Haptic text variables
        self.haptic_text = None
        self.haptic_text_start_time = None
        self.haptic_text_duration = 3.0
        self.haptic_text_alpha = 0.0
    
    def _count_fingers(self, hand_landmarks):
        """Count the number of extended fingers in a hand (excluding thumb)."""
        # Finger tip landmarks (excluding thumb)
        finger_tips = [
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
            mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
            mp.solutions.hands.HandLandmark.PINKY_TIP
        ]
        
        # Corresponding PIP (second joint) landmarks
        finger_pips = [
            mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
            mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
            mp.solutions.hands.HandLandmark.PINKY_PIP
        ]
        
        extended_fingers = 0
        
        for i in range(len(finger_tips)):
            tip_y = hand_landmarks.landmark[finger_tips[i]].y
            pip_y = hand_landmarks.landmark[finger_pips[i]].y
            # Finger is extended if tip is above PIP joint
            if tip_y < pip_y - 0.01:  # Small threshold for stability
                extended_fingers += 1
        
        return extended_fingers
    
    def _is_thumbs_down(self, hand_landmarks):
        """Checks for a proper thumbs-down gesture."""
        # Get thumb landmarks
        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP]
        thumb_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP]
        wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
        
        # For a proper thumbs down, we need:
        # 1. Thumb pointing downward (tip below IP joint)
        # 2. Other fingers curled in a fist
        # 3. Hand orientation should be more vertical than horizontal
        
        # Check that thumb is extended downward (tip below IP joint)
        if thumb_tip.y > thumb_ip.y:
            # Check that other fingers are curled (fist-like position)
            finger_tips = [
                mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                mp.solutions.hands.HandLandmark.PINKY_TIP
            ]
            finger_pips = [
                mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
                mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
                mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
                mp.solutions.hands.HandLandmark.PINKY_PIP
            ]
            
            # Check if other fingers are curled (not extended)
            fingers_curled = True
            for i in range(len(finger_tips)):
                tip_y = hand_landmarks.landmark[finger_tips[i]].y
                pip_y = hand_landmarks.landmark[finger_pips[i]].y
                if tip_y < pip_y - 0.01:  # Finger is extended (with small threshold)
                    fingers_curled = False
                    break
            
            # Check hand orientation - for thumbs down, the hand should be more vertical
            # Compare wrist to finger positions to determine hand orientation
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
            
            # Calculate hand width vs height
            hand_width = abs(index_tip.x - pinky_tip.x)
            hand_height = abs(thumb_tip.y - wrist.y)
            
            # Check that thumb is the lowest point of the hand
            # Get all finger tips to compare with thumb
            all_finger_tips = [
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP],
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP],
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP],
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
            ]
            
            # Check if thumb is lower than all other finger tips
            thumb_is_lowest = True
            for finger_tip in all_finger_tips:
                if thumb_tip.y <= finger_tip.y:  # Thumb is not lower than this finger
                    thumb_is_lowest = False
                    break
            
            # For thumbs down, hand should be more vertical than horizontal
            # Also ensure thumb is pointing significantly downward and is the lowest point
            if (hand_height > hand_width * 0.8 and  # Hand is more vertical
                thumb_tip.y > wrist.y + 0.05 and    # Thumb is significantly below wrist
                thumb_is_lowest and                  # Thumb is the lowest point
                fingers_curled):
                return True
        
        return False
    
    def _is_middle_finger(self, hand_landmarks):
        """Checks for a middle finger gesture (flipping off)."""
        # Check if middle finger is extended while others are curled
        middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
        
        # Middle finger should be extended (tip above PIP)
        if middle_tip.y >= middle_pip.y:
            return False
        
        # Check that other fingers (except thumb) are curled
        finger_tips = [
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
            mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
            mp.solutions.hands.HandLandmark.PINKY_TIP
        ]
        finger_pips = [
            mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
            mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
            mp.solutions.hands.HandLandmark.PINKY_PIP
        ]
        
        # Check if other fingers are curled (not extended)
        for i in range(len(finger_tips)):
            tip_y = hand_landmarks.landmark[finger_tips[i]].y
            pip_y = hand_landmarks.landmark[finger_pips[i]].y
            if tip_y < pip_y:  # Finger is extended
                return False
        
        return True
    
    def _handle_gesture_activation(self, finger_count):
        """Handle gesture-based feature activation (only enables features)."""
        current_time = time.time()
        
        # Check if we're in cooldown period
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return
        
        # Map finger count to features - each feature has its own hand position
        gesture_features = {
            1: 'pose_detection',      # 1 finger = pose detection
            2: 'ascii_effect',        # 2 fingers = ASCII effect
            3: 'face_mesh',           # 3 fingers = face mesh
            4: 'face_blackout',       # 4 fingers = face blackout
            5: 'face_overlay'         # 5 fingers = face overlay
        }
        
        if finger_count in gesture_features:
            feature_name = gesture_features[finger_count]
            
            # Only enable the feature if it's not already enabled
            if feature_name in self.ui_manager.checkboxes and not self.ui_manager.checkboxes[feature_name]['checked']:
                self.ui_manager.checkboxes[feature_name]['checked'] = True
                
                # Special handling for ASCII effect - start timer
                if feature_name == 'ascii_effect':
                    self.ascii_start_time = current_time
                    self._show_haptic_text(f"{self.ui_manager.checkboxes[feature_name]['label']} enabled for 10 seconds!")
                else:
                    self._show_haptic_text(f"{self.ui_manager.checkboxes[feature_name]['label']} enabled!")
                
                self.last_gesture_time = current_time
                print(f"Gesture {finger_count} fingers: {self.ui_manager.checkboxes[feature_name]['label']} enabled")
            else:
                # Feature is already enabled, show a different message
                self._show_haptic_text(f"{self.ui_manager.checkboxes[feature_name]['label']} is already enabled!")
                self.last_gesture_time = current_time
                print(f"Gesture {finger_count} fingers: {self.ui_manager.checkboxes[feature_name]['label']} already enabled")
    
    def _handle_gesture_deactivation(self):
        """Handle thumbs-down gesture to deactivate finger-controllable features."""
        current_time = time.time()
        
        # Check if we're in cooldown period
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return
        
        # Features that can be deactivated by thumbs down
        deactivatable_features = ['pose_detection', 'ascii_effect', 'face_mesh', 'face_blackout', 'face_overlay']
        
        deactivated_count = 0
        for feature_name in deactivatable_features:
            if feature_name in self.ui_manager.checkboxes and self.ui_manager.checkboxes[feature_name]['checked']:
                self.ui_manager.checkboxes[feature_name]['checked'] = False
                deactivated_count += 1
        
        if deactivated_count > 0:
            self._show_haptic_text(f"Deactivated {deactivated_count} features!")
            self.last_gesture_time = current_time
            print(f"Thumbs down: Deactivated {deactivated_count} features")
    
    def _show_haptic_text(self, message):
        """Shows a haptic text message."""
        self.haptic_text = message
        self.haptic_text_start_time = time.time()
        self.haptic_text_alpha = 0.0
    
    def process_gestures(self, hand_results, current_gesture, current_finger_count, thumbs_down_detected):
        """Process hand gestures and return updated state."""
        thumbs_down_detected = False
        middle_finger_detected = False
        current_gesture = None
        current_finger_count = None
        
        # --- Hand detection and gesture tracking ---
        if (self.ui_manager.checkboxes['hand_detection']['checked'] and 
            hand_results and hand_results.multi_hand_landmarks):
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Check for middle finger gesture
                if self._is_middle_finger(hand_landmarks):
                    middle_finger_detected = True
                    # Cancel any ongoing gesture detection when middle finger is detected
                    current_gesture = None
                    self.gesture_start_time = None
                    self.current_gesture = None
                # Check for thumbs down gesture
                elif self._is_thumbs_down(hand_landmarks):
                    thumbs_down_detected = True
                    # Cancel any ongoing gesture detection when thumbs-down is detected
                    current_gesture = None
                    self.gesture_start_time = None
                    self.current_gesture = None
                else:
                    # Count fingers for gesture detection if not in special gesture positions
                    finger_count = self._count_fingers(hand_landmarks)
                    current_finger_count = finger_count
                    if 1 <= finger_count <= 5:
                        current_gesture = finger_count
        
        # --- Gesture duration logic ---
        now = time.time()
        if current_gesture is not None and not thumbs_down_detected and not middle_finger_detected:
            # Check if the gesture corresponds to an already enabled feature
            gesture_features = {
                1: 'pose_detection',
                2: 'ascii_effect', 
                3: 'face_mesh',
                4: 'face_blackout',
                5: 'face_overlay'
            }
            
            feature_name = gesture_features.get(current_gesture)
            feature_already_enabled = (feature_name in self.ui_manager.checkboxes and 
                                    self.ui_manager.checkboxes[feature_name]['checked'])
            
            # Don't show progress for already enabled features
            if feature_already_enabled:
                # Reset gesture tracking for already enabled features
                self.gesture_start_time = None
                self.current_gesture = None
                current_gesture = None
            else:
                # Normal gesture tracking for disabled features
                if self.gesture_start_time is None or self.current_gesture != current_gesture:
                    self.gesture_start_time = now
                    self.current_gesture = current_gesture
                elif now - self.gesture_start_time >= self.gesture_duration:
                    # Gesture held for required duration, trigger activation
                    self._handle_gesture_activation(current_gesture)
                    # Reset gesture tracking to stop progress bar
                    self.gesture_start_time = None
                    self.current_gesture = None
                    current_gesture = None  # Also reset the local variable
        elif thumbs_down_detected:
            # Handle thumbs-down deactivation
            self._handle_gesture_deactivation()
            # Reset gesture tracking after deactivation
            self.gesture_start_time = None
            self.current_gesture = None
            current_gesture = None
        else:
            # No gesture detected or special gestures detected, reset tracking
            self.gesture_start_time = None
            self.current_gesture = None
        
        # --- ASCII effect with auto-disable ---
        if self.ui_manager.checkboxes['ascii_effect']['checked']:
            # Initialize ascii_start_time if it's None (checkbox was checked directly)
            if self.ascii_start_time is None:
                self.ascii_start_time = now
            
            # Check if ASCII effect should auto-disable
            elapsed_time = now - self.ascii_start_time
            if elapsed_time >= self.ascii_duration:
                # Auto-disable ASCII effect after duration
                self.ui_manager.checkboxes['ascii_effect']['checked'] = False
                self.ascii_start_time = None
                self._show_haptic_text("ASCII effect auto-disabled!")
        else:
            # Reset ASCII timer when not in ASCII mode
            self.ascii_start_time = None
        
        return current_gesture, current_finger_count, thumbs_down_detected, middle_finger_detected 