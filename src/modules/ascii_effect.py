import cv2
import numpy as np
from .effect_base import EffectProcessor

class ASCIIEffect(EffectProcessor):
    """Converts video frames to ASCII art representation."""
    
    def __init__(self):
        self.font_scale = 0.4
        self.cell_size = 8
        self.chars = "@%#*+=-:. "
    
    def process(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Convert frame to ASCII art representation.
        
        Args:
            frame: Input frame as numpy array
            **kwargs: Additional parameters (unused)
            
        Returns:
            ASCII art frame as numpy array
        """
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