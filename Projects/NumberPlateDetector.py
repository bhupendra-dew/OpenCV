# Issme thoda error a raha hai  

import cv2
import numpy as np

######################################

frameWidth = 640
frameHeigth = 480

nPlateCascade = cv2.CascadeClassifier("no.plate.jpg")
minArea = 200
color = (255, 255, 255)

######################################

while True :
    success, img = cv2.read(nPlateCascade)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)
    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea :
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            imgRoi = img[y : y + h, x : x + w]
            cv2.imshow("ROI", imgRoi)

    cv2.imshow("Result", img)

    k = cv2.waitKey(1) & 0xFF == ord()
    if k == 27 :
        break

        