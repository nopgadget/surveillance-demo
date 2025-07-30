import cv2
import numpy as np
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import time

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

class FaceSwapOptimizer:
    """Optimization class for face swap operations."""
    def __init__(self):
        self.cache = {}
        self.last_face_detection = None
        self.last_landmarks = None
        self.frame_count = 0
        self.last_processed_frame = None
        self.last_face_positions = []
        # Use more workers for better performance
        self.executor = ThreadPoolExecutor(max_workers=6)
        
    def get_cached_or_compute(self, key: str, compute_func):
        """Cache results to avoid redundant computations."""
        if key not in self.cache:
            self.cache[key] = compute_func()
        return self.cache[key]
    
    def should_process_frame(self, skip_interval=1) -> bool:
        """Determine if current frame should be processed based on skip interval."""
        return self.frame_count % skip_interval == 0
    
    def should_process_frame_smart(self, current_faces=None):
        """Smart frame processing - skip if faces haven't moved significantly."""
        if current_faces is None:
            return True
            
        # Always process first few frames
        if self.frame_count < 5:
            return True
            
        # If no faces detected, skip processing
        if len(current_faces) == 0:
            return False
            
        # Check if faces have moved significantly
        current_positions = [(face.left(), face.top(), face.right(), face.bottom()) for face in current_faces]
        
        if len(self.last_face_positions) == 0:
            self.last_face_positions = current_positions
            return True
            
        # Check if any face has moved significantly (more than 10 pixels)
        for i, (curr_pos, last_pos) in enumerate(zip(current_positions, self.last_face_positions)):
            if i >= len(last_pos):
                return True
            if (abs(curr_pos[0] - last_pos[0]) > 10 or 
                abs(curr_pos[1] - last_pos[1]) > 10 or
                abs(curr_pos[2] - last_pos[2]) > 10 or
                abs(curr_pos[3] - last_pos[3]) > 10):
                self.last_face_positions = current_positions
                return True
                
        # Faces haven't moved much, skip processing
        return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)

class OptimizedFaceSwapEffect(EffectProcessor):
    """Advanced face swap effect with optimizations from face-swap-experiment.py."""
    
    def __init__(self, source_face_path: str = "img/kevin-hart.jpg"):
        self.source_face_path = source_face_path
        self.optimizer = FaceSwapOptimizer()
        self.source_data = None
        self.detector = None
        self.predictor = None
        self._initialize_face_detection()
        self._preprocess_source_face()
        
    def _initialize_face_detection(self):
        """Initialize dlib face detection components."""
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        except ImportError:
            print("Warning: dlib not available. Face swap will be disabled.")
            self.detector = None
            self.predictor = None
        except FileNotFoundError:
            print("Warning: shape_predictor_68_face_landmarks.dat not found. Face swap will be disabled.")
            self.detector = None
            self.predictor = None
    
    def _preprocess_source_face(self):
        """Preprocess source face once and cache results."""
        if self.detector is None or self.predictor is None:
            return
            
        try:
            img = cv2.imread(self.source_face_path)
            if img is None:
                print(f"Error: Could not load source image {self.source_face_path}")
                return
                
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(img_gray)

            faces = self.detector(img_gray, 1)

            if len(faces) == 0:
                print("No faces detected in source image")
                return

            face = faces[0]
            landmarks = self.predictor(img_gray, face)
            
            # Extract landmarks more efficiently using list comprehension
            landmark_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
            points = np.array(landmark_points, np.int32)
            
            convexhull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, convexhull, (255,))
            face_image_1 = cv2.bitwise_and(img, img, mask=mask)

            # Precompute triangles more efficiently
            rect = cv2.boundingRect(convexhull)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(landmark_points)
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)

            indexes_triangles = []
            for t in triangles:
                pt1, pt2, pt3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])
                
                # More efficient point matching using numpy operations
                index_pt1 = np.where((points == pt1).all(axis=1))[0]
                index_pt2 = np.where((points == pt2).all(axis=1))[0]
                index_pt3 = np.where((points == pt3).all(axis=1))[0]
                
                if len(index_pt1) > 0 and len(index_pt2) > 0 and len(index_pt3) > 0:
                    triangle = [index_pt1[0], index_pt2[0], index_pt3[0]]
                    indexes_triangles.append(triangle)

            self.source_data = {
                'img': img,
                'img_gray': img_gray,
                'mask': mask,
                'face_image_1': face_image_1,
                'landmark_points': landmark_points,
                'points': points,
                'convexhull': convexhull,
                'indexes_triangles': indexes_triangles
            }
            
        except Exception as e:
            print(f"Error preprocessing source face: {e}")
            self.source_data = None
    
    def _process_triangle_optimized(self, triangle_data):
        """Process a single triangle with optimized operations."""
        triangle_index, source_data, landmark_points2, img2, img2_new_face = triangle_data
        
        # Precompute triangle points
        tr1_pt1 = source_data['landmark_points'][triangle_index[0]]
        tr1_pt2 = source_data['landmark_points'][triangle_index[1]]
        tr1_pt3 = source_data['landmark_points'][triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        
        # Early exit for invalid triangles
        if w <= 0 or h <= 0:
            return None
            
        cropped_triangle = source_data['img'][y: y + h, x: x + w]
        if cropped_triangle.size == 0:
            return None
            
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        points = np.array([
            [tr1_pt1[0] - x, tr1_pt1[1] - y],
            [tr1_pt2[0] - x, tr1_pt2[1] - y],
            [tr1_pt3[0] - x, tr1_pt3[1] - y]
        ], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, (255,))
        cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)

        tr2_pt1 = landmark_points2[triangle_index[0]]
        tr2_pt2 = landmark_points2[triangle_index[1]]
        tr2_pt3 = landmark_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2
        
        # Early exit for invalid triangles
        if w <= 0 or h <= 0:
            return None
            
        cropped_triangle2 = img2[y: y + h, x: x + w]
        if cropped_triangle2.size == 0:
            return None

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([
            [tr2_pt1[0] - x, tr2_pt1[1] - y],
            [tr2_pt2[0] - x, tr2_pt2[1] - y],
            [tr2_pt3[0] - x, tr2_pt3[1] - y]
        ], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, (255,))
        cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask=cropped_tr2_mask)

        src_points = points.astype(np.float32)
        dst_points = points2.astype(np.float32)
        
        if src_points.shape[0] == 3 and dst_points.shape[0] == 3:
            try:
                M = cv2.getAffineTransform(src_points, dst_points)
                warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                
                warped_mask = cv2.warpAffine(cropped_tr1_mask, M, (w, h))
                
                warped_mask_3d = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
                warped_mask_3d = warped_mask_3d.astype(np.float32) / 255.0
                
                return {
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'warped_triangle': warped_triangle,
                    'warped_mask_3d': warped_mask_3d
                }
            except cv2.error:
                return None
        
        return None
    
    def _process_face_swap(self, img2):
        """Optimized face swap processing with multithreading support."""
        if self.source_data is None or self.detector is None or self.predictor is None:
            return img2
            
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Adaptive face detection - use lower sensitivity if we already have faces
        upsample_factor = 1 if len(self.optimizer.last_face_positions) > 0 else 2
        faces2 = self.detector(img2_gray, upsample_factor)
        
        # Smart frame processing - skip if faces haven't moved significantly
        if not self.optimizer.should_process_frame_smart(faces2):
            # Return cached result if available
            if self.optimizer.last_processed_frame is not None:
                return self.optimizer.last_processed_frame
            return img2
        
        if len(faces2) == 0:
            self.optimizer.last_processed_frame = img2
            return img2
        
        result = img2
        for face2 in faces2:
            # Early exit if face is too small (likely false positive)
            rect = face2
            if rect.width() < 30 or rect.height() < 30:  # Less restrictive size check
                continue
            
            img2_new_face = np.zeros_like(img2)
            
            landmarks2 = self.predictor(img2_gray, face2)
            
            # Extract landmarks more efficiently using list comprehension
            landmark_points2 = [(landmarks2.part(n).x, landmarks2.part(n).y) for n in range(68)]
            
            img2_face_mask = np.zeros_like(img2_gray)
            points2 = np.array(landmark_points2, np.int32)
            convexhull2 = cv2.convexHull(points2)
            cv2.fillConvexPoly(img2_face_mask, convexhull2, (255,))

            # Adaptive triangle processing - use fewer triangles for smaller faces
            face_size = max(rect.width(), rect.height())
            if face_size < 100:
                # Use every 3rd triangle for small faces
                triangle_indices = self.source_data['indexes_triangles'][::3]
            elif face_size < 200:
                # Use every 2nd triangle for medium faces
                triangle_indices = self.source_data['indexes_triangles'][::2]
            else:
                # Use all triangles for large faces
                triangle_indices = self.source_data['indexes_triangles']
            
            # Process triangles with multithreading
            triangle_data_list = [
                (triangle_index, self.source_data, landmark_points2, img2, img2_new_face)
                for triangle_index in triangle_indices
            ]
            
            # Process triangles in parallel
            futures = [self.optimizer.executor.submit(self._process_triangle_optimized, data) 
                      for data in triangle_data_list]
            
            # Collect results and apply them
            for future in futures:
                result_data = future.result()
                if result_data:
                    x, y, w, h = result_data['x'], result_data['y'], result_data['w'], result_data['h']
                    warped_triangle = result_data['warped_triangle']
                    warped_mask_3d = result_data['warped_mask_3d']
                    
                    triangle_area = img2_new_face[y:y + h, x:x + w]
                    alpha = warped_mask_3d
                    beta = 1.0 - alpha
                    blended = cv2.addWeighted(triangle_area, 1.0, warped_triangle, 1.0, 0)
                    blended = blended * beta + warped_triangle * alpha
                    img2_new_face[y:y + h, x:x + w] = blended.astype(np.uint8)

            # Optimized blending
            img2_new_face_gray = cv2.cvtColor(img2_new_face, cv2.COLOR_BGR2GRAY)
            _, face_mask = cv2.threshold(img2_new_face_gray, 1, 255, cv2.THRESH_BINARY)

            face_mask = cv2.GaussianBlur(face_mask, (3, 3), 0)
            face_mask = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2BGR)
            face_mask = face_mask.astype(np.float32) / 255.0

            background = img2.astype(np.float32)
            new_face = img2_new_face.astype(np.float32)

            result = background * (1.0 - face_mask) + new_face * face_mask
            result = result.astype(np.uint8)

            # Optimized seamless cloning with error handling
            try:
                face_mask_gray = cv2.cvtColor(face_mask.astype(np.uint8) * 255, cv2.COLOR_BGR2GRAY)
                _, face_mask_binary = cv2.threshold(face_mask_gray, 127, 255, cv2.THRESH_BINARY)

                moments = cv2.moments(face_mask_binary)
                if moments["m00"] != 0:
                    center_x = int(moments["m10"] / moments["m00"])
                    center_y = int(moments["m01"] / moments["m00"])
                    center = (center_x, center_y)
                    
                    # Ensure center is within image bounds
                    h, w = img2.shape[:2]
                    center_x = max(0, min(center_x, w-1))
                    center_y = max(0, min(center_y, h-1))
                    center = (center_x, center_y)
                    
                    result = cv2.seamlessClone(img2_new_face, img2, face_mask_binary, center, cv2.NORMAL_CLONE)
            except cv2.error:
                # Fall back to simple blending if seamless cloning fails
                pass
        
        # Cache the result for reuse
        self.optimizer.last_processed_frame = result
        return result
    
    def process(self, frame: np.ndarray, face_mesh_results=None, **kwargs) -> np.ndarray:
        """Apply optimized face swap effect to the frame."""
        if self.source_data is None:
            return frame
            
        self.optimizer.frame_count += 1
        
        # Smart processing with caching and motion detection
        return self._process_face_swap(frame)
    
    def cleanup(self):
        """Clean up resources."""
        if self.optimizer:
            self.optimizer.cleanup() 