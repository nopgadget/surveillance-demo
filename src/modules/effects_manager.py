from ..effect_processor import ASCIIEffect, FaceOverlayEffect, FaceBlackoutEffect

class EffectsManager:
    """Handles all effects management and application."""
    
    def __init__(self, ui_manager):
        self.ui_manager = ui_manager
        
        # Initialize effects
        self.effects = {
            'ascii': ASCIIEffect(),
            'face_overlay': FaceOverlayEffect(ui_manager.assets['face_overlay']),
            'face_blackout': FaceBlackoutEffect()
        }
    
    def apply_effects(self, frame, face_mesh_results):
        """Apply all active effects to the frame."""
        # Apply ASCII effect
        if self.ui_manager.checkboxes['ascii_effect']['checked']:
            frame = self.effects['ascii'].process(frame)
        
        # Apply face overlay
        if self.ui_manager.checkboxes['face_overlay']['checked']:
            frame = self.effects['face_overlay'].process(frame, face_mesh_results=face_mesh_results)
        
        # Apply face blackout
        if self.ui_manager.checkboxes['face_blackout']['checked']:
            frame = self.effects['face_blackout'].process(frame, face_mesh_results=face_mesh_results)
        
        return frame 