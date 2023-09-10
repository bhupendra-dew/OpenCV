# WebCam 

'''
import cv2
import webbrowser

cap = cv2.VideoCapture(0)
detector = cv2.QRCodeDetector()

while True:
    _, img = cap.read()
    data, bbox, _ = detector.detectAndDecode(img)
    if data:
        a = data
        break

    cv2.imshow("QR_CODE_Scanner", img)
    k = cv2.waitKey(1) & 0xFF 
    if k == 27:
        break

b = webbrowser.open(str(a))
cap.release()
cv2.destroyAllWindows()

'''

import cv2
import webbrowser

cap = cv2.VideoCapture(0)
detector = cv2.QRCodeDetector()

while True:
    _, img = cap.read()
    data, bbox, _ = detector.detectAndDecode(img)
    if data:
        a = data
        break

    cv2.imshow("QR_CODE_Scanner", img)
    k = cv2.waitKey(1) & 0xFF 
    if k == 27:
        break

b = webbrowser.open(str(a))
cap.release()
cv2.destroyAllWindows()
