# Homogeneous filter - Each output pixel is the mean of its kernel neighbours
# In image processing, a kernel, convolution matrix, or mask is a small matrix. It is used for blurring, sharpening, embossing, edge detection, and more.
# As in one-dimensional siginals, images also can be filtered with various low-pass filters(LPF), high-pass filters(HPF) etc..
# LPF helps in removing nosises, bluring the images.
# HPF filters helps in finding the edges in the images.
# 2d convolution reduces the picture quality by shortening the matrix value ie from 6*6 to 3*3 there by reducing the quality of the image, that why the sharpness of the image is gone.

''' 
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('opencv-logo.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((5, 5), np.float32)/25
dat = cv2.filter2D(img, -1, kernel)

titles = ['image', '2D Convolution']
images = [img, dat]

for i in range(i):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

'''
# Method 1 The blur method or the averaging method.
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('opencv-logo.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((5, 5), np.float32)/25
dat = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img, (5,5));

titles = ['image', '2D Convolution', 'blur']
images = [img, dat, blur]

for i in range(3):
    plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
'''

# Method 2 Gaussian filter method
# Gaussian filter method is nothing but using different-weight-kerenl, in both x and y direction.
# Here the pixels located on the side have lower weigth than the pixels located on the centre.
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Salt and pepper .jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((5, 5), np.float32)/25
dat = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img, (5,5));
gblur = cv2.GaussianBlur(img, (5,5), 0)

titles = ['image', '2D Convolution', 'blur', 'Gaussian Blur']
images = [img, dat, blur, gblur]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
'''
# Method 3 Median filter method
# Median fiter is somethig that replace each pixels value with the median of its neighbouring pixels. This method is great when dealing with 'salt and pepper noise'.
# Salt and peper nosie is a foem of noise sometimes seen on the images. It is also known as impulse noise.
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Salt and pepper .jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((5, 5), np.float32)/25
dat = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img, (5,5));
gblur = cv2.GaussianBlur(img, (5,5), 0)
median = cv2.medianBlur(img, 3)

titles = ['image', '2D Convolution', 'blur', 'Gaussian Blur', 'Median']
images = [img, dat, blur, gblur, median]

for i in range(5):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
'''
# Method 4 Bilateral Filter 
# It is useful in keeing the borders blur free while keeping the edges sharp.

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Salt and pepper .jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((5, 5), np.float32)/25
dat = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img, (5,5));
gblur = cv2.GaussianBlur(img, (5,5), 0)
median = cv2.medianBlur(img, 3)
bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)

titles = ['image', '2D Convolution', 'blur', 'Gaussian Blur', 'Median', 'Bilateral Filter']
images = [img, dat, blur, gblur, median, bilateralFilter]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()