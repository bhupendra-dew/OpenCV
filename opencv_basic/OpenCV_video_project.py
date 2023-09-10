#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      bhupendra
#
# Created:     27-06-2023
# Copyright:   (c) bhupe 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------
# Code for switching on camera using opencv and then closing it with pressing q key.
'''
import cv2
cap = cv2.VideoCapture(0);
while(True):
    ret, frame = cap.read()                      # Here frames are being capured
    cv2.imshow("Frame", frame)                   # Frames are being shown  "Frame" is the name given to the window.
    if cv2.waitKey(1) & 0xFF == ord('q'):        # For qutting the camera
        break
cap.release()
cv2.destroyAllWindows()
'''
# Code to change the colours in grey
'''
import cv2
cap = cv2.VideoCapture(0);
while(True):
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Frame", grey)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''
# Code for opening a video file for the computer and checking if its path directory is right or not.
'''
import cv2
cap = cv2.VideoCapture("Megamind.avi");
print(cap.isOpened())
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''
# properties for reading get() frame width as well as height
'''
import cv2
cap = cv2.VideoCapture("Megamind.avi");
print(cap.isOpened())
while(cap.isOpened()):
    ret, frame = cap.read()
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''
# Video saving class for caputring the video from camera.
'''
import cv2
cap = cv2.VideoCapture(0);
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
print(cap.isOpened())
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret ==  True:
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
'''