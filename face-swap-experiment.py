import cv2
import numpy as np
import dlib # conda install -c conda-forge dlib
from typing import List, Tuple, Optional, Dict, Any
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Set to True for webcam, False for image input
USE_WEBCAM = True

# Image paths
SOURCE_IMAGE = "img/cox.jpg"  # Source face to swap from
TARGET_IMAGE = "img/jack-black.jpg"  # Target image to swap onto (when USE_WEBCAM = False)

# Set to True to use seamless cloning, False for simple blending
USE_SEAMLESS_CLONE = True

# Performance optimization flags
ENABLE_CACHING = True
ENABLE_EARLY_EXIT = True
ENABLE_FRAME_SKIPPING = True
ENABLE_MULTITHREADING = True
ENABLE_MEMORY_OPTIMIZATION = True
FRAME_SKIP_INTERVAL = 1  # Process every Nth frame
MAX_WORKERS = 4  # Number of worker threads for triangle processing

class FaceSwapOptimizer:
    def __init__(self):
        self.cache = {}
        self.last_face_detection = None
        self.last_landmarks = None
        self.frame_count = 0
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS) if ENABLE_MULTITHREADING else None
        
    def extract_index_nparray(self, nparray):
        """Extract index from numpy array."""
        index = None
        for num in nparray[0]:
            index = num
            break
        return index
    
    def get_cached_or_compute(self, key: str, compute_func):
        """Cache results to avoid redundant computations."""
        if not ENABLE_CACHING or key not in self.cache:
            self.cache[key] = compute_func()
        return self.cache[key]
    
    def should_process_frame(self) -> bool:
        """Determine if current frame should be processed based on skip interval."""
        if not ENABLE_FRAME_SKIPPING:
            return True
        return self.frame_count % FRAME_SKIP_INTERVAL == 0
    
    def cleanup(self):
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)

# Initialize optimizer
optimizer = FaceSwapOptimizer()

def preprocess_source_face():
    """Preprocess source face once and cache results."""
    img = cv2.imread(SOURCE_IMAGE)
    if img is None:
        print(f"Error: Could not load source image {SOURCE_IMAGE}")
        exit()
        
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)

    detector = dlib.get_frontal_face_detector()  # type: ignore
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # type: ignore

    faces = detector(img_gray, 1)

    if len(faces) == 0:
        print("No faces detected in source image")
        exit()

    face = faces[0]
    landmarks = predictor(img_gray, face)
    
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

    return {
        'img': img,
        'img_gray': img_gray,
        'mask': mask,
        'face_image_1': face_image_1,
        'landmark_points': landmark_points,
        'points': points,
        'convexhull': convexhull,
        'indexes_triangles': indexes_triangles,
        'detector': detector,
        'predictor': predictor
    }

def process_triangle_optimized(triangle_data):
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

def process_face_swap(img2, source_data):
    """Optimized face swap processing with multithreading support."""
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_new_face = np.zeros_like(img2)
    
    faces2 = source_data['detector'](img2_gray, 1)
    
    if len(faces2) == 0:
        return img2
    
    result = img2
    for face2 in faces2:
        # Early exit if face is too small (likely false positive)
        if ENABLE_EARLY_EXIT:
            rect = face2
            if rect.width() < 50 or rect.height() < 50:
                continue
        
        img2_new_face = np.zeros_like(img2)
        
        landmarks2 = source_data['predictor'](img2_gray, face2)
        
        # Extract landmarks more efficiently using list comprehension
        landmark_points2 = [(landmarks2.part(n).x, landmarks2.part(n).y) for n in range(68)]
        
        img2_face_mask = np.zeros_like(img2_gray)
        points2 = np.array(landmark_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)
        cv2.fillConvexPoly(img2_face_mask, convexhull2, (255,))

        # Process triangles with optional multithreading
        if ENABLE_MULTITHREADING and optimizer.executor:
            # Prepare triangle data for parallel processing
            triangle_data_list = [
                (triangle_index, source_data, landmark_points2, img2, img2_new_face)
                for triangle_index in source_data['indexes_triangles']
            ]
            
            # Process triangles in parallel
            futures = [optimizer.executor.submit(process_triangle_optimized, data) 
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
        else:
            # Sequential processing (original method)
            for triangle_index in source_data['indexes_triangles']:
                result_data = process_triangle_optimized((triangle_index, source_data, landmark_points2, img2, img2_new_face))
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

        # Optimized seamless cloning
        if USE_SEAMLESS_CLONE:
            face_mask_gray = cv2.cvtColor(face_mask.astype(np.uint8) * 255, cv2.COLOR_BGR2GRAY)
            _, face_mask_binary = cv2.threshold(face_mask_gray, 127, 255, cv2.THRESH_BINARY)

            moments = cv2.moments(face_mask_binary)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                center = (center_x, center_y)
                
                result = cv2.seamlessClone(img2_new_face, img2, face_mask_binary, center, cv2.NORMAL_CLONE)
    
    return result

# Preprocess source face once
source_data = preprocess_source_face()

WINDOW_NAME = "Face Swap"
WINDOW_MAX_WIDTH = 1280
WINDOW_MAX_HEIGHT = 720
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_MAX_WIDTH, WINDOW_MAX_HEIGHT)

def resize_with_aspect_ratio(image, max_width, max_height):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        while True:
            ret, img2 = cap.read()
            if not ret:
                break
            
            optimizer.frame_count += 1
            
            # Frame skipping for performance
            if not optimizer.should_process_frame():
                img2_resized = resize_with_aspect_ratio(img2, WINDOW_MAX_WIDTH, WINDOW_MAX_HEIGHT)
                cv2.imshow(WINDOW_NAME, img2_resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
                
            result = process_face_swap(img2, source_data)
            result_resized = resize_with_aspect_ratio(result, WINDOW_MAX_WIDTH, WINDOW_MAX_HEIGHT)
            
            cv2.imshow(WINDOW_NAME, result_resized)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        optimizer.cleanup()
else:
    # Image input mode
    img2 = cv2.imread(TARGET_IMAGE)
    if img2 is None:
        print(f"Error: Could not load target image {TARGET_IMAGE}")
        exit()
        
    result = process_face_swap(img2, source_data)
    result_resized = resize_with_aspect_ratio(result, WINDOW_MAX_WIDTH, WINDOW_MAX_HEIGHT)

    cv2.imshow(WINDOW_NAME, result_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    optimizer.cleanup()