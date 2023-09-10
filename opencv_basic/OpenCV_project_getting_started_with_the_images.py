#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      bhupe
#
# Created:     26-06-2023
# Copyright:   (c) bhupe 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------
# Shows lena image for 10 seconds with its data in matrices form and then destroy itself after 10 seconds.
'''
import cv2
img = cv2.imread("lena.jpg", 0)
print(img)
cv2.imshow("image", img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
'''
# Shows lena image for 10 seconds with its data in matrices form and then destroy itself after 10 seconds with creating a new copy of lena image.
'''
import cv2
img = cv2.imread("lena.jpg", 1)
print(img)
cv2.imshow("image", img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite("lena_copy.png", img)
'''
# If someone presses esc key then image shoud be destroyed or is someone preses s key then the image gets saved.
'''
import cv2
img = cv2.imread("lena.jpg", 0)
cv2.imshow("image", img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite("lena_copy.png", img)
    cv2.destroyAllWindows()
'''
