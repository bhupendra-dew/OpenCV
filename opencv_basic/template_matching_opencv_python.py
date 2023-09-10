# Template matching is a method of searching and finding the location of a template image inside a larger image.
import cv2
import numpy as np
img = cv2.imread('messi5.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('face_messi5.jpg', 0)

w, h = template.shape[::-1]

res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
print(res)
threshold = 0.95; # yaha par hum sabse brightest point ko nikal rahe hai,nahi to agar humara no. agar chota ho wo sare points ko hume bata de ga jisme use utne intensity mil rahi hai.
loc = np.where(res >= threshold)
print(loc)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()