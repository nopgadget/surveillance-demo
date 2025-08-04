import cv2
import numpy as np
from abc import ABC, abstractmethod

class EffectProcessor(ABC):
    """Abstract base class for all video effects."""
    
    @abstractmethod
    def process(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Process a frame and return the modified frame.
        
        Args:
            frame: Input frame as numpy array
            **kwargs: Additional parameters for the effect
            
        Returns:
            Processed frame as numpy array
        """
        pass 