# basic and simple arethmatic operations done on an image using opencv
'''
import numpy as np
import cv2
img = cv2.imread('messi5.jpg')
print(img.shape)      # returns a tuple of number of rows, columns, and channels
print(img.size)       # returns Total number of pixels is accessed
print(img.dtype)      # returns Image Datatype is obtained
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))
ball = img[280:340,330:390]
img[273:333, 100:160] = ball
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Adding two images

import numpy as np
import cv2
img = cv2.imread('messi5.jpg')
img2 = cv2.imread('opencv-logo.png')
print(img.shape)      # returns a tuple of number of rows, columns, and channels
print(img.size)       # returns Total number of pixels is accessed
print(img.dtype)      # returns Image Datatype is obtained
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))
ball = img[280:340,330:390]
img[273:333, 100:160] = ball
img = cv2.resize(img, (512,512))
img2 = cv2.resize(img2, (512,512))
dst = cv2.add(img, img2);
dst = cv2.addWeighted(img, .9, img2, .1, 0); # Here messi image is dominant. 
cv2.imshow('image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()