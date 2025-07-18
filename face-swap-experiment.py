import cv2
import numpy as np
import dlib # conda install -c conda-forge dlib


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

img = cv2.imread("img/dwayne.png")
img2 = cv2.imread("img/kevin-hart.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray)

detector = dlib.get_frontal_face_detector()  # type: ignore
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # type: ignore
faces = detector(img_gray, 1)
for face in faces:
    landmarks = predictor(img_gray, face)
    landmark_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points.append((x, y))

    points = np.array(landmark_points,np.int32)
    convexhull = cv2.convexHull(points)
    cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
    cv2.fillConvexPoly(mask, convexhull, (255,))
    face_image_1 = cv2.bitwise_and(img, img, mask=mask)

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    
    indexes_triangles = []

    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2],t[3])
        pt3 = (t[4],t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1,index_pt2,index_pt3]
            indexes_triangles.append(triangle)

        cv2.line(img,pt1,pt2, (0,0,255),2)
        cv2.line(img,pt2,pt3, (0,0,255),2)
        cv2.line(img,pt1,pt3, (0,0,255),2)

        # Face 2
        faces2 = detector(img2_gray, 1)
        for face in faces2:
            landmarks = predictor(img2_gray, face)
            landmark_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmark_points.append((x, y))

                #cv2.circle(img2, (x,y), 3, (0,255,0), -1)

# Triangulation of second face, from first face delaunay triangulation

for triangle_index in indexes_triangles:
    pt1 = landmark_points[triangle_index[0]]
    pt2 = landmark_points[triangle_index[1]]
    pt3 = landmark_points[triangle_index[2]]

    cv2.line(img2, pt1, pt2, (0,0,255),1)
    cv2.line(img2, pt2, pt3, (0,0,255),1)
    cv2.line(img2, pt1, pt3, (0,0,255),1)



cv2.imshow("img", img)
cv2.imshow("img2",img2)

cv2.waitKey(0)
cv2.destroyAllWindows()