# Image Gradient - an image gradient is a direction change in the intensity or the color in an image.
# There are three types of methods (they are different gradient functions which uses different mathematical operations so to produce the req. images.)- laplacian methods, sobel x method and sobel y method.
# Laplacian Method - It calculates the laplacian derivatives.
# Sobel Method - It is joint, gaussian and differentiation operations.

# Method 1 - Laplacian gradient method
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg', cv2.IMREAD_GRAYSCALE)
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3) # cv2.CV_64F is just a data type. Here we are using a 64 bit float due to the negative slope induced by transforming the image.
# Converting the 64 bit value to the unsigned 8 bit integer which is suitable for our output.
lap = np.uint8(np.absolute(lap))

titles = ['image', 'Laplacian']
images = [img, lap]
for i in  range(2) :
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
'''
# Method 2 Soble x and soble y method Or soble gradient method.

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg', cv2.IMREAD_GRAYSCALE)
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3) # cv2.CV_64F is just a data type. Here we are using a 64 bit float due to the negative slope induced by transforming the image.
# Converting the 64 bit value to the unsigned 8 bit integer which is suitable for our output.
lap = np.uint8(np.absolute(lap))
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

# Combining sobelX and sobelY 
sobelCombined = cv2.bitwise_or(sobelX, sobelY)

titles = ['image', 'Laplacian', 'SobelX', 'SobelY', 'Sobel Combined']
images = [img, lap, sobelX, sobelY, sobelCombined]
for i in  range(5) :
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()