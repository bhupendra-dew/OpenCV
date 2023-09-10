# Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in the images.
# The Canny edge detection algorithm is composed of 5 steps. - Noise reduction, Gradient calculation, Non-maximum suppression, Double threshold, Edge Tracking by Hysteresis.
# 1st step apply gaussian filter to the image in order to remove the noise.
# 2nd step find the intensity gradient of the image.
# 3rd step to apply the non maximum suppression of the image to get rid of the spurrious response over the edge detection.
# 4th step to apply double threshold do determine the potential edges.
# 5th step is to track edges using hysteresis, ie. to fianlise the detection of the edges by supressing the edges that are weak or are not connected to the representative.

import cv2 
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg', 0)
canny = cv2.Canny(img, 100, 200)

titles = ['image', 'canny']
images = [img, canny]
for i in range(2) :
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
