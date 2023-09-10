# Is code ke jarye hum mobile phone ke camera ko use kaer sakete hai as a seconday camera, bus hume ek app install karna pade ga jis ka name hai IP Webcam 
# IP Webcam ka IPV4 address browsre mai type karne se uska web page load ho jata hai and then hume us video ka inspection me se jo link mile use open cv ke normal video capture code me input kar dena hai.
# Note - Dono devices ek hi network se connected hone chiye.

'''
import cv2

capture =  cv2.VideoCapture("http://192.168.1.3:8080/video")

while (True) :
    _, frame = capture.read()
    cv2.imshow('Live Stream', frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
'''

# Now wahi video ko hum gray scale me convert kar rahe hai.

import cv2

capture =  cv2.VideoCapture("http://192.168.1.3:8080/video")

while (True) :
    _, frame = capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imshow('Live Stream', gray)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()