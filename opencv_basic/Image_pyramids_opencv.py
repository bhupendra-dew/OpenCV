# by using this we can create image of different resolution.(downscaling the image quality)
# Pyramid, or pyramid representation, is a type of multi-scale signal representation in which a signal or an image is subject to repeated smoothing and subsampling.
# Isme ye ho raha hai ki humare paas jo original image hai hum usko har bhar blur and subsample karke uski resolution ko half karte hai upne desired need tak.
# there are two types of open pyramids in opencv - Gaussian pyramid, The Laplacian pyramid

# Gaussian pyramid - Repeated filtering and subsampaling of an image. There are two functions in the gaussian pyramid they are - pyrDown and pyrUp
# Here we are using both the pyrDown and pyrUp methods.
# Note onece we use the pyrdown method the inoformation or the details of the in=mage is been lost so by again upscaling that same image will give us less detailed image than it should be.
'''
import cv2
import numpy as np
img = cv2.imread('lena.jpg')

lr1 = cv2.pyrDown(img) # Reduces the resolution of the image.
lr2 = cv2.pyrDown(lr1) # Further more reduces the resolution of the image.
ur = cv2.pyrUp(img) # Increases the resolution of the image.

cv2.imshow('Oringinal Image', img)
cv2.imshow('Pyramid_Down1', lr1)
cv2.imshow('Pyramid_Down2', lr2)
cv2.imshow('Pyramid_Up', ur)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# using for loop for scaling down the image resolution
'''
import cv2
import numpy as np
img = cv2.imread('lena.jpg')
layer = img.copy()
gp = [layer]

for i in range(6):
    layer = cv2.pyrDown(layer)
    gp.append(layer)
    cv2.imshow(str(i), layer)

cv2.imshow('Original image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# Laplacian pyramid - A level in Laplacian Pyramid is formed by the difference between that level in the gaussian Pyramid and the expanded version of its upper level in the Gaussian Pyramid.

import cv2
import numpy as np

img = cv2.imread('lena.jpg')
layer = img.copy()
gp = [layer]

for i in range(6):
    layer = cv2.pyrDown(layer)
    gp.append(layer)
    #cv2.imshow(str(i), layer)

layer = gp[5]
cv2.imshow('upper level Gaussian Pyramid', layer)
lp = [layer]

for i in range(5, 0, -1):
    gaussian_extended = cv2.pyrUp(gp [i])
    laplacian = cv2.subtract(gp[i-1], gaussian_extended)
    cv2.imshow(str(i), laplacian)

cv2.imshow('original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# The use of creating laplacian and gaussian pyramids is that they help us to blend the images and the reconstruction of the images.
