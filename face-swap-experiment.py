import cv2
import numpy as np
import dlib # conda install -c conda-forge dlib

img = cv2.imread("img/dwayne.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    cv2.fillConvexPoly(mask, convexhull, (255,))
    face_image_1 = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("img", img)
cv2.imshow("mask", mask)
cv2.imshow("face_image_1", face_image_1)

cv2.waitKey(0)
cv2.destroyAllWindows()