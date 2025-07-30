from ..effect_processor import ASCIIEffect, FaceBlackoutEffect, OptimizedFaceSwapEffect

class EffectsManager:
    """Handles all effects management and application."""
    
    def __init__(self, ui_manager):
        self.ui_manager = ui_manager
        
        # Initialize effects
        self.effects = {
            'ascii': ASCIIEffect(),
            'face_blackout': FaceBlackoutEffect(),
            'face_swap': OptimizedFaceSwapEffect("img/kevin-hart.jpg")
        }
    
    def apply_effects(self, frame, face_mesh_results):
        """Apply all active effects to the frame."""
        # Apply ASCII effect
        if self.ui_manager.checkboxes['ascii_effect']['checked']:
            frame = self.effects['ascii'].process(frame)
        
        # Apply face swap
        if self.ui_manager.checkboxes.get('face_swap', {}).get('checked', False):
            frame = self.effects['face_swap'].process(frame, face_mesh_results=face_mesh_results)
        
        # Apply face blackout
        if self.ui_manager.checkboxes['face_blackout']['checked']:
            frame = self.effects['face_blackout'].process(frame, face_mesh_results=face_mesh_results)
        
        return frame
    
    def cleanup(self):
        """Clean up resources for effects that need it."""
        if 'face_swap' in self.effects:
            self.effects['face_swap'].cleanup() 