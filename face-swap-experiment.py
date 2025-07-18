import cv2
import numpy as np
import dlib # conda install -c conda-forge dlib

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

img = cv2.imread("img/musk.jpg")
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
landmark_points = []
for n in range(0, 68):
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    landmark_points.append((x, y))

points = np.array(landmark_points, np.int32)
convexhull = cv2.convexHull(points)
cv2.fillConvexPoly(mask, convexhull, (255,))

face_image_1 = cv2.bitwise_and(img, img, mask=mask)

rect = cv2.boundingRect(convexhull)
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(landmark_points)
triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype=np.int32)

indexes_triangles = []
for t in triangles:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])

    index_pt1 = np.where((points == pt1).all(axis=1))
    index_pt1 = extract_index_nparray(index_pt1)

    index_pt2 = np.where((points == pt2).all(axis=1))
    index_pt2 = extract_index_nparray(index_pt2)
    
    index_pt3 = np.where((points == pt3).all(axis=1))
    index_pt3 = extract_index_nparray(index_pt3)

    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        triangle = [index_pt1, index_pt2, index_pt3]
        indexes_triangles.append(triangle)

cap = cv2.VideoCapture(0)

while True:
    ret, img2 = cap.read()
    if not ret:
        break
        
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_new_face = np.zeros_like(img2)
    
    faces2 = detector(img2_gray, 1)
    
    if len(faces2) > 0:
        for face2 in faces2:
            img2_new_face = np.zeros_like(img2)
            
            landmarks2 = predictor(img2_gray, face2)
            landmark_points2 = []
            for n in range(0, 68):
                x = landmarks2.part(n).x
                y = landmarks2.part(n).y
                landmark_points2.append((x, y))

            img2_face_mask = np.zeros_like(img2_gray)
            points2 = np.array(landmark_points2, np.int32)
            convexhull2 = cv2.convexHull(points2)
            cv2.fillConvexPoly(img2_face_mask, convexhull2, (255,))

            for triangle_index in indexes_triangles:
                tr1_pt1 = landmark_points[triangle_index[0]]
                tr1_pt2 = landmark_points[triangle_index[1]]
                tr1_pt3 = landmark_points[triangle_index[2]]
                triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                rect1 = cv2.boundingRect(triangle1)
                (x, y, w, h) = rect1
                cropped_triangle = img[y: y + h, x: x + w]
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
                cropped_triangle2 = img2[y: y + h, x: x + w]

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
                        
                        triangle_area = img2_new_face[y:y + h, x:x + w]
                        
                        warped_mask_3d = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
                        warped_mask_3d = warped_mask_3d.astype(np.float32) / 255.0
                        
                        alpha = warped_mask_3d
                        beta = 1.0 - alpha
                        blended = cv2.addWeighted(triangle_area, 1.0, warped_triangle, 1.0, 0)
                        blended = blended * beta + warped_triangle * alpha
                        img2_new_face[y:y + h, x:x + w] = blended.astype(np.uint8)
                    except cv2.error:
                        continue

            img2_new_face_gray = cv2.cvtColor(img2_new_face, cv2.COLOR_BGR2GRAY)
            _, face_mask = cv2.threshold(img2_new_face_gray, 1, 255, cv2.THRESH_BINARY)

            face_mask = cv2.GaussianBlur(face_mask, (3, 3), 0)
            face_mask = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2BGR)
            face_mask = face_mask.astype(np.float32) / 255.0

            background = img2.astype(np.float32)
            new_face = img2_new_face.astype(np.float32)

            result = background * (1.0 - face_mask) + new_face * face_mask
            result = result.astype(np.uint8)

            face_mask_gray = cv2.cvtColor(face_mask.astype(np.uint8) * 255, cv2.COLOR_BGR2GRAY)
            _, face_mask_binary = cv2.threshold(face_mask_gray, 127, 255, cv2.THRESH_BINARY)

            moments = cv2.moments(face_mask_binary)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                center = (center_x, center_y)
                
                result = cv2.seamlessClone(img2_new_face, img2, face_mask_binary, center, cv2.NORMAL_CLONE)
            
            img2 = result
    else:
        result = img2

    cv2.imshow("Face Swap", result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()