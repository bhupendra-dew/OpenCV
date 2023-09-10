'''
import matplotlib.pyplot as plt
import cv2
import numpy as np

image = cv2.imread('Road.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [(0, height), (width/2, height/2), (width, height)]

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255, ) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    mask_image = cv2.bitwise_and(img, mask)
    return mask_image

masked_image = region_of_interest(image, np.array([region_of_interest_vertices], np.int32))

plt.imshow(masked_image)
plt.show()
'''
'''
import matplotlib.pyplot as plt
import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255, ) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    mask_image = cv2.bitwise_and(img, mask)
    return mask_image

image = cv2.imread('Road1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [(0, height), (width/2, height/2), (width, height)]

gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
canny_image = cv2.Canny(gray_image, 100, 200)

masked_image = region_of_interest(image, np.array([region_of_interest_vertices], np.int32))

plt.imshow(canny_image)
plt.show()
'''
'''
import matplotlib.pyplot as plt
import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    mask_image = cv2.bitwise_and(img, mask)
    return mask_image

def draw_the_lines(img, lines):
    copy_img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness = 4)
        
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img
 
image = cv2.imread('Road3.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [(0, height), 
                               (width/2, height/2), 
                               (width, height)]

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny_image = cv2.Canny(gray_image, 100, 200)

masked_image = region_of_interest(canny_image, 
                                  np.array([region_of_interest_vertices], 
                                           np.int32))

lines = cv2.HoughLinesP(masked_image, 
                        rho = 6, theta = np.pi/60, 
                        threshold = 60, 
                        lines = np.array([]), 
                        minLineLength = 40,  
                        maxLineGap = 25)

image_with_lines = draw_the_lines(image, lines)
plt.imshow(image_with_lines)
plt.show()
'''

# Python OpenCV - Canny() Function
'''
import cv2
img = cv2.imread('Road.jpg')   # Reading the image

# setting parameter values
t_lower = 50    # Lower threshold
t_upper = 150   # Upper thyreshold

# Applying the canny edge filter
edge = cv2.Canny(img, t_lower, t_upper)

cv2.imshow('Original', img)
cv2.imshow('Edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Canny() function with Aperture_size
'''
import cv2

img = cv2.imread('Road5.jpg')

t_lower = 100
t_upper = 200
aperture_size = 5
edge = cv2.Canny(img, t_lower, t_upper, apertureSize = aperture_size)
cv2.imshow('original', img)
cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Canny() Function with L2Gradient
'''
import cv2

img = cv2.imread('Road1.jpg')
t_lower = 100
t_upper = 200
aperture_size = 5
L2Gradient = True

edge = cv2.Canny(img, t_lower, t_upper, L2g
radient = L2Gradient)

cv2.imshow('Origianl', img)
cv2.imshow('Edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Canny() function with body Aperture size and L2gradient
'''
import cv2

img = cv2.imread('mrpumpkin.jpeg')

t_lower = 100
t_upper = 200
aperture_size = 5
L2Gradient = True

edge = cv2.Canny(img, t_lower, t_upper,
                 apertureSize = aperture_size,
                 L2gradient = L2Gradient)

cv2.imshow("Original", img)
cv2.imshow("Edge", edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''