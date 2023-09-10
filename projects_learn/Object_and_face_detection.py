'''
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()
    # ret will return a true value if the frame exits othewise False
    into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # changing the color format from the BGR to HSV. This will be used to create the mask
    L_limit = np.array([98, 50, 50])      # setting the blue lower limt
    U_limit = np.array([139, 255, 255])   # setting the blue upper limt

    b_mask = cv2.inRange(into_hsv, L_limit, U_limit)
    # creating the mask using the inRange() function, this will produce an image where the coolor of the objects falling in the range will return white and the rest will be black
    blue = cv2.bitwise_and(frame, frame, mask = b_mask)
    # this will give the color to the mask.

    cv2.imshow('Original', frame)        # To display the original frame
    cv2.imshow('Blue Detector', blue)    # to diaplay the blue object output

    if cv2.waitKey(1) == 27:
        break

    # this function will be triggered when the ESC key is pressed and the while loop will terminate ad so will the program
cap.release()
cv2.destroyAllWindows() 
'''
# Python Program to detect the edges of an image using OpenCV | Sobel edge detection method
'''
# Python program to Edge detection using OpenCV in Pyhton using Sobel edge detection and laplacian method
import cv2
import numpy as np

# Capture livestream video content from camera 0
cap = cv2.VideoCapture(0)

while(1):
    # Take each frame 
    _, frame = cap.read()
    # Convert to HSV for simpler calculations 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Calculation of Sobelx
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize = 5)
    # Calculation of Sobely
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize = 5)
    # Calculation of Laplacian
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)

    cv2.imshow('Sobelx', sobelx)
    cv2.imshow('Sobely', sobely)
    cv2.imshow('Lapalcaian', laplacian)
    
    k = cv2.waitKey(5) & 0xFF 
    if k == 27:
        break
cv2.destroyAllWindows()
# release the frame
cap.release()
'''
# Corner detection with Harris Corner Detection method using the OpenCV

# Python program to illustrate corner detection with Harris Corner Detection Method
# Organizing imports
'''
import cv2
import numpy as np
# Path to input image specified and image is loaded with the imread command
image = cv2.imread('temple.jpeg')
# Convert the input image into the grayscale color space
operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# modify the data type setting the 32 - bit floating point
operatedImage = np.float32(operatedImage)
# Apply the cv2.cornerHarris method to detect the corners with appropriate values as input parameters 
dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)
# Results are marked through the dilated corneres
dest = cv2.dilate(dest, None)
# Reverting back to the original image, with optimal threshold value
image[dest > 0.01 * dest.max()] = [ 0, 0, 255]
# the window showing output image with corners
cv2.imshow('Image Border', image)
# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
'''

# OpenCV Program for face detection
'''
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

    cv2.imshow("IMAGE", img)

    k = cv2.waitKey(0) & 0xFF 
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
'''