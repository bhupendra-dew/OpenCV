# Isme hum shapes ko detet karke unka name bata rahe hai.


import numpy as np
import cv2

# Adding a tackbar 
'''
import cv2 as cv

def nothing(x):
    print(x)

img = cv.imread('shapes.jpg')
cv.namedWindow('image')
cv.createTrackbar('CP', 'image', 10, 500, nothing)

switch = 'colour/gray'
cv.createTrackbar(switch, 'image', 0, 1, nothing)

while(1):
    img = cv.imread('shapes.jpg')
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

# Increasing the size of the image using pyramid method.

img = cv2.imread('Shapes.png')
'''
ur1 = cv2.pyrUp(img) # Increases the resolution of the image.
ur = cv2.pyrUp(ur1)  # ek aur bar resolution increase kar rahe hai.

cv2.imshow('Oringinal Image', img)
cv2.imshow('Pyramid_Up', ur)
'''
# Image gradient ko use kar rahe hai

from matplotlib import pyplot as plt
ur = img

lap = cv2.Laplacian(ur, cv2.CV_64F, ksize=3) # cv2.CV_64F is just a data type. Here we are using a 64 bit float due to the negative slope induced by transforming the image.
# Converting the 64 bit value to the unsigned 8 bit integer which is suitable for our output.
lap = np.uint8(np.absolute(lap))
sobelX = cv2.Sobel(ur, cv2.CV_64F, 1, 0, ksize=3)
sobelY = cv2.Sobel(ur, cv2.CV_64F, 0, 1, ksize=3)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

# Combining sobelX and sobelY 
sobelCombined = cv2.bitwise_or(sobelX, sobelY)

titles = ['image', 'Laplacian', 'SobelX', 'SobelY', 'Sobel Combined']
images = [ur, lap, sobelX, sobelY, sobelCombined]
for i in  range(5) :
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# Shape Detection ka code continoue yaha se hai

imgGray = cv2.cvtColor(ur, cv2.COLOR_BGR2GRAY)    # Image ko gray color me convert kar rahe hai
_, thresh = cv2.threshold(imgGray, 240, 255, cv2.THRESH_BINARY)  # image ki jo matrices value ko binary format me convert kar rahe hai
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # jo bhi vertices hume mile hai usme wo points dalega and un points ko join karega.

for contour in  contours :
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    cv2.drawContours(ur, [approx], 0, (0, 0 ,0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 3:
        cv2.putText(ur, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    elif len(approx) == 4:
        x1, y1, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        print(aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            cv2.putText(sobelCombined, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0))
            cv2.putText(ur, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
        else:
            cv2.putText(sobelCombined, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0))
            cv2.putText(ur, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    elif len(approx) == 5:
        cv2.putText(sobelCombined, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0))
        cv2.putText(ur, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    elif len(approx) == 10:
        cv2.putText(sobelCombined, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0))
        cv2.putText(ur, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    else :
        cv2.putText(sobelCombined, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0))
        cv2.putText(ur, "circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))

cv2.imshow('shapes', ur)
cv2.imshow('shapes1', sobelCombined)
cv2.waitKey(0)
cv2.destroyAllWindows()
