# Drawing line on the lena photo.
'''
import numpy as np
import cv2
img = cv2.imread('lena.jpg', 1)
img = cv2.line(img, (0,0), (255, 255), (255, 0, 0), 10) # this is done as to add a line in the photo of blue color with width 10. It is in bgr format.
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Drawing line and arrow in the photo
'''
import numpy as np
import cv2
img = cv2.imread('lena.jpg', 1)
img = cv2.line(img, (0,0), (255, 255), (255, 0, 0), 5)
img = cv2.arrowedLine(img, (0,255), (255,255), (255,0,0),10)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Drawing a rectangle
'''
import numpy as np
import cv2
img = cv2.imread('lena.jpg', 1)
img = cv2.line(img, (0,0), (255, 255), (255, 0, 0), 5)
img = cv2.arrowedLine(img, (0,255), (255,255), (255,0,0),10)
img =  cv2.rectangle(img, (384,0), (510,128), (0,0,255), 0)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Drawing a Circle as well as putting text into the image
'''
import numpy as np
import cv2
img = cv2.imread('lena.jpg', 1)
img = cv2.line(img, (0,0), (255, 255), (255, 0, 0), 5)
img = cv2.arrowedLine(img, (0,255), (255,255), (255,0,0),10)
img = cv2.rectangle(img, (384,0), (510,128), (0,0,255), 0)
img = cv2.circle(img, (477, 63), 63, (0, 255, 0), -1)
font = cv2.FONT_HERSHEY_COMPLEX
img = cv2.putText(img, 'OpenCv', (10, 500), font, 4, (0, 0, 255), 5, cv2.LINE_AA)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Using numpy for reading an image insted of imread function.
'''
import numpy as np
import cv2
#img = cv2.imr9ead('lena.jpg', 1)
img = np.zeros([512, 512, 3], np.uint8)
img = cv2.line(img, (0,0), (255, 255), (255, 0, 0), 5)
img = cv2.arrowedLine(img, (0,255), (255,255), (255,0,0),10)
img = cv2.rectangle(img, (384,0), (510,128), (0,0,255), 0)
img = cv2.circle(img, (477, 63), 63, (0, 255, 0), -1)
font = cv2.FONT_HERSHEY_COMPLEX
img = cv2.putText(img, 'OpenCv', (10, 500), font, 4, (0, 0, 255), 5, cv2.LINE_AA)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
