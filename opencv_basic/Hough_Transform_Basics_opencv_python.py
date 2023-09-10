# Hough Transform Basics - The hough transform is a popular technique to detect any shape, if you can represent that shape in a mathematical forml.
# It can detect the shape even if it is broken or distorted a little bit.
# Haugh transform basics is the iport method jo hume humre import pixels ko jaida waitage deta hai.
# Hough Transform Basics - A line in the image space can be expressed with two variables. Fo example;
# 1. In the cartesian coordinate system   y = mx + c
# 2. In the polar coordinate system       xcos(theta) + ysin(theta) = r

# Algorithms involved in the Hough Transform.
# 1. Edge detection, eg. using the canny edge detector.
# 2. Mapping of edge points to the Hough Space and storage in an accumulator.
# 3. Interpretation of the accumulator to the yield lines of the infinte length. Teh interpretation is done bhy thresholding and possibly other constraints.
# 4. Conversion of the infinte lines to finite lines.

# OpenCV implents two kinds  of Hough Line Transforms
# * The Standard Hough Transform (HoughLines method)
# * The Probabibilistic Hough Line Trasform (HoughLinesP method)


# Here Standard Hough Transform(HoughLines method) is Used......

'''
import cv2
import numpy as np

img = cv2.imread('sudoku.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
cv2.imshow('Canny Edge Image', edges)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

'''
# Hough line method takes in few arguments and here are those arguments whith their meaning.
'''
lines = cv.HoughLines(image, rho, theta, threshold)

images : source image
lines : Output of the vector of lines. Each line is represented by a 2 or 3 element vector (rho, theta) or (rho, theta, votes).
        rho is the distance from the coordinate origin (0, 0)(top-left corner of the image).
        theta is the line rotation angle in radians.
        votes is the value of accumulator.
rho : Distace resolution of the accumulator in the pixels.
theta : Angle resolution of the accumulator in radians.
threshold : Accumulator threshold parameter. Only those lines are returned that get enough votes(>threshold).
'''
# Continue of the above code
'''
for line in lines :
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
    x1 = int(x0 + 1000 * (-b))
    # y1 stroes the rounded off value of (r * sin(theta) + 1000 * cos(theta))
    y1 = int(y0 + 1000 * (a))
    # x2 stores the rounded off value of (r * cos(theta) + 1000 * sin(theta))
    x2 = int(x0 - 1000 * (-b))
    # y2 stores the rounded off value of (r * sin(theta) - 1000 * cos(theta))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2) 

cv2.imshow('Image', img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()
 
'''

# Here the Probabilistic Hough Line Transform (HoughLinesP method)

import cv2
import numpy as np

img = cv2.imread('Road3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

cv2.imshow('Edges', edges)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength = 100, maxLineGap = 10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('Image', img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()