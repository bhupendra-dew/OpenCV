
# OpenCV program to perform the Edge Detection in the real time
# import libraries of python OpenCV
# where its functionality resides
'''
import cv2
# np is an alias pointing to numpy library
import numpy as np
# capture frames from a camera
cap = cv2.VideoCapture('video (540p).avi')
# loop runs if capturing has been initialized
while(1) :
    # reads frmaes from a camera
    ret, frame = cap.read()
    # converting BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of red color in HSV
    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])
    # create a red HSV color boundary and threshold HSV image
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # Bitwise-AND mask and oringinal image
    res = cv2.bitwise_and(frame, frame, mask = mask)
    # Display an original image
    cv2.imshow('Original', frame)
    # finds the edges in the input image and marks them in the output map edges.
    edges = cv2.Canny(frame, 100, 200)
    # Display edges in the frame
    cv2.imshow('Edges', edges)
    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
# Close the window 
cap.release()
# De-allocate any associated memory useage
cv2.destroyAllWindows()
'''
'''
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# defining the canny detector function, here weak_th and stron_th are threshold for double.
def Canny_detector(img, weak_th = None, strong_th = None):
    # conersion of image of grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Noise reduction step
    img = cv2.GaussianBlur(img, (5, 5), 1.4)
    # calculating the gradients
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
    # conversion of the cartesian coordinates to the polar
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True)
    # setting the minimum and maximum thressholds, for double thresholdoing.
    mag_max = np.max(mag)
    if not weak_th:weak_th = mag_max * 0.1
    if not strong_th:strong_th = mag_max * 0.5
    # getting the dimensions of the input image
    height, width = img.shape
    # Looping through every pixel of the grayscale image
    for i_x in range(width):
        for i_y in range(height):

            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            # selecting the neighbours of the target pixel, according to the gradient direction.
            # In the x - axis direction
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
            
            # top right (diagonal - 1) diection
            elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

            # In y - axis direction 
            elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y - 1
                neighb_2_x, neighb_2_y = i_x, i_y - 1

            # top left (diagonal - 2) direction
            elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y - 1 

            # now it restarta the cycle
            elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # Non - maximum supression step
            if width > neighb_1_x >= 0 and height > neighb_1_y >= 0 :
                if mag[i_y, i_x] <mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x] = 0 
                    continue
            
            if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                if mag[i_y, i_x] < mag[neighb_1_y, neighb_2_x]:
                    mag[i_y, i_x] = 0

    weak_ids = np.zeros_like(img)
    strong_ids = np.zeros_like(img)
    ids = np.zeros_like(img)

    # double thresholding step
    for i_x in range(width):
        for i_y in range(height):

            grad_mag = mag[i_y, i_x]

            if grad_mag < weak_th:
                mag[i_y, i_x] = 0
            elif strong_th > grad_mag >= weak_th:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2

    # finally returing the magnitude of the gradients of the edges
    return mag

frame = cv2.imread('temple.jpeg')

# calling the designed function for finding edges
canny_image = Canny_detector(frame)

# Displaying the input and output image
plt.figure()
f, plots = plt.subplots(2, 1)
plots[0].imshow(frame)
plots[1].imshow(canny_image)
plt.show()
'''