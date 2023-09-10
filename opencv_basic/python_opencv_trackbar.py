# Example 1
# Trackbars in opencv. Helps in chaning some value dinamically in an  image.
'''
import numpy as np
import cv2 as cv

def nothing(x):
    print(x)

# creating a black image, a window

img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('image')
cv.createTrackbar('B', 'image', 0, 255, nothing)
cv.createTrackbar('G', 'image', 0, 255, nothing)
cv.createTrackbar('R', 'image', 0, 255, nothing)

while(1):
    cv.imshow('image', img)
    k = cv.waitKey(1) & 0xFF
    if k == 27 :
        break

    b = cv.getTrackbarPos('B', 'image')
    g = cv.getTrackbarPos('G', 'image')
    r = cv.getTrackbarPos('R', 'image')

    img[:] = [b, g, r]

cv.destroyAllWindows()
'''
# Adding a switch to the trackbar.
'''
import numpy as np
import cv2 as cv

def nothing(x):
    print(x)

# creating a black image, a window

img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('image')
cv.createTrackbar('B', 'image', 0, 255, nothing)
cv.createTrackbar('G', 'image', 0, 255, nothing)
cv.createTrackbar('R', 'image', 0, 255, nothing)

switch = '0 : OFF\n 1 : ON'
cv.createTrackbar(switch, 'image', 0, 1, nothing)

while(1):
    cv.imshow('image', img)
    k = cv.waitKey(1) & 0xFF
    if k == 27 :
        break

    b = cv.getTrackbarPos('B', 'image')
    g = cv.getTrackbarPos('G', 'image')
    r = cv.getTrackbarPos('R', 'image')
    s = cv.getTrackbarPos(switch, 'image')

    if s == 0 :
        img [:] = 0
    else:
        img[:] = [b, g, r]
    img[:] = [b, g, r]

cv.destroyAllWindows()
'''

# Example 2 Coloured to grey scale image.
'''
import numpy as np
import cv2 as cv

def nothing(x):
    print(x)

img = cv.imread('lena.jpg')
cv.namedWindow('image')
cv.createTrackbar('CP', 'image', 10, 500, nothing)

switch = 'colour/gray'
cv.createTrackbar(switch, 'image', 0, 1, nothing)

while(1):
    img = cv.imread('lena.jpg')
    pos = cv.getTrackbarPos('CP', 'image')
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, str(pos), (50, 150), font, 4, (0, 0 , 255))
    k = cv.waitKey(1) & 0xFF
    if k == 27 :
        break 
    s = cv.getTrackbarPos(switch, 'image')
    
    if s == 0 :
        pass
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    img = cv.imshow('image', img)

cv.destroyAllWindows()

'''