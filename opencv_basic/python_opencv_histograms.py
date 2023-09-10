# Histogram is a graph or a plot which gives us the overall idea about the intensity distribution of an image. 
# Histogram is just another way of looking the image. Histrogram can give us the idea about the constrast, brightness, intensity distribution etc..
# Method 1 Finding the hisogram using the matplotlib.
'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('lena.jpg')
#img = np.zeros((200, 200), np.uint8)
#cv.rectangle(img, (0, 100), (200, 200), (255), -1)
#cv.rectangle(img, (0, 50), (100, 100), (127), -1)

b, g, r = cv.split(img)

cv.imshow('img', img)
cv.imshow('b', b)
cv.imshow('g', g)
cv.imshow('r', r)

plt.hist(b.ravel(), 256, [0, 256])
plt.hist(g.ravel(), 256, [0, 256])
plt.hist(r.ravel(), 256, [0, 256])
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
'''
# Method 2 calcHist method

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('lena.jpg')

hist  = cv.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
