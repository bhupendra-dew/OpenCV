'''
import cv2
import numpy as np 

img = cv2.imread('Shapes.png')
image = cv2.resize(img, (700, 600))

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower = np.array([50, 100, 100])
upper = np.array([70, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

cv2.imshow("Image", image)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
'''
'''
# Python programs to find
# unique HSV code for color

# Importing the libraries openCV & numpy
import cv2
import numpy as np

# Get green color
green = np.uint8([[[0, 255, 0]]])

# Convert Green color to Green HSV
hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)

# Print HSV Value for Green color
print(hsv_green)

# Make python sleep for unlimited time
cv2.waitKey(0)
'''
