import cv2
import numpy as np
from abc import ABC, abstractmethod

class EffectProcessor(ABC):
    @abstractmethod
    def process(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        pass

class ASCIIEffect(EffectProcessor):
    def __init__(self):
        self.font_scale = 0.4
        self.cell_size = 8
        self.chars = "@%#*+=-:. "
    
    def process(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        out_img = np.zeros_like(frame)
        
        for y in range(0, h, self.cell_size):
            for x in range(0, w, self.cell_size):
                cell = gray[y:y+self.cell_size, x:x+self.cell_size]
                if cell.size == 0:
                    continue
                avg = int(np.mean(cell))
                char_idx = int((avg / 255) * (len(self.chars) - 1))
                char = self.chars[char_idx]
                
                green_intensity = int((avg / 255) * 255) + 100
                color = (0, green_intensity, 0)
                
                cv2.putText(
                    out_img, char, (x, y + self.cell_size),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 1, cv2.LINE_AA
                )
        return out_img

class FaceOverlayEffect(EffectProcessor):
    def __init__(self, face_overlay_img: np.ndarray):
        self.face_overlay = face_overlay_img
    
    def process(self, frame: np.ndarray, face_mesh_results=None, **kwargs) -> np.ndarray:
        """Overlay a generic face image on detected faces using the convex hull of FaceMesh landmarks."""
        if face_mesh_results is None or not face_mesh_results.multi_face_landmarks:
            return frame
        
        if self.face_overlay is None:
            return frame
            
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            points = []
            h, w, _ = frame.shape
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                points.append([x, y])
            points = np.array(points, dtype=np.int32)
            if points.shape[0] > 0:
                hull = cv2.convexHull(points)
                
                # Get the bounding rectangle of the face
                x, y, w_face, h_face = cv2.boundingRect(hull)
                
                # Resize the face overlay to match the detected face size, scaled up 2x
                face_overlay_resized = cv2.resize(self.face_overlay, (w_face * 2, h_face * 2))
                
                # Create a mask for the face region
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)
                
                # Extract the region of interest (accounting for 2x scaled overlay)
                # Center the larger overlay over the detected face
                x_offset = w_face // 2
                y_offset = h_face // 2
                roi_x = max(0, x - x_offset)
                roi_y = max(0, y - y_offset)
                roi_w = min(w_face * 2, frame.shape[1] - roi_x)
                roi_h = min(h_face * 2, frame.shape[0] - roi_y)
                
                roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                mask_roi = mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                
                if roi.shape[0] > 0 and roi.shape[1] > 0 and face_overlay_resized.shape[0] > 0 and face_overlay_resized.shape[1] > 0:
                    # Handle alpha channel if present
                    if face_overlay_resized.shape[2] == 4:
                        # Extract alpha channel and normalize
                        alpha = face_overlay_resized[:, :, 3] / 255.0
                        alpha = np.clip(alpha, 0, 1)
                        
                        # Blend the face overlay with the original frame
                        for c in range(3):
                            roi[:, :, c] = (face_overlay_resized[:, :, c] * alpha + 
                                           roi[:, :, c] * (1 - alpha)).astype(np.uint8)
                    else:
                        # No alpha channel, use the mask
                        mask_alpha = mask_roi / 255.0
                        for c in range(3):
                            roi[:, :, c] = (face_overlay_resized[:, :, c] * mask_alpha + 
                                           roi[:, :, c] * (1 - mask_alpha)).astype(np.uint8)
                    
                    # Put the blended region back
                    frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = roi
                    
        return frame

class FaceBlackoutEffect(EffectProcessor):
    def process(self, frame: np.ndarray, face_mesh_results=None, **kwargs) -> np.ndarray:
        """Blackout detected faces using the convex hull of FaceMesh landmarks."""
        if face_mesh_results is None or not face_mesh_results.multi_face_landmarks:
            return frame
            
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            points = []
            h, w, _ = frame.shape
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                points.append([x, y])
            points = np.array(points, dtype=np.int32)
            if points.shape[0] > 0:
                hull = cv2.convexHull(points)
                
                # Create a black mask for the face region
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)
                
                # Apply black color to the face region
                black_color = (0, 0, 0)  # BGR format: black
                frame[mask > 0] = black_color
                    
        return frame 