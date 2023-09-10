# Here we are taking about the morphological opreations like erosion, dilation opening and closing methods etc..
# Morphological operations are some simple operations based on the image shape. It is normally performed on the binary image. There are two things required while doing morphological operaations.
# First is the original image and second is the kernel.
# A Kernel tells how to change the value of any given pixel by combining it with different amount of neighbouring pixels.
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('smarties.png', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

kernel =  np.ones((5, 5), np.uint8)

dilation = cv2.dilate(mask, kernel, iterations = 5)
erosion = cv2.erode(mask, kernel, iterations = 1)
titles = ['image', 'mask', 'dilation', 'erosions']
images = [img, mask, dilation, erosion]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
'''
# Opening and closing , imroves in remoing black spots from the image.

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('smarties.png', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

kernel =  np.ones((5, 5), np.uint8)

dilation = cv2.dilate(mask, kernel, iterations = 5)
erosion = cv2.erode(mask, kernel, iterations = 1)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
th = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)

titles = ['image', 'mask', 'dilation', 'erosions', 'opening', 'closing', 'mg', 'th']
images = [img, mask, dilation, erosion, opening, closing, mg, th]

for i in range(8):
    plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()