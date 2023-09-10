# Contours are the curves joining all the continous points along the boundary which are having the same color or density.
# Contours are useful tools for shape analogy or object detection or objet recognition.
# For better accuracy binary images are used for finding the contour.
# contours is a python list of all the contours in the image. Each individual contour is a Numpy array of (x, y) coordinates of boundary points of the object.
# hierarchy is the optional output vector which is containing the formation about image topology.

# NOW Applying the threshold

import numpy as np
import cv2

img = cv2. imread('opencv-logo.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print('Number of contours = ' + str(len(contours)))
print(contours[0])

cv2.drawContours(img, contours, -1, (0,255,0), 5)

cv2.imshow('Image', img)
cv2.imshow('Image GRAY', imgray)

cv2.waitKey(0)
cv2.destroyAllWindows()