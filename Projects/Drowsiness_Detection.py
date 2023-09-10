'''
import cv2
import lib
import pyttsx3
from scipy.spatial import distance

# INITIALIZING THE pyttsx3 SO THAT
# ALERT AUDIO MESSAGE CAN BE DELIVERED
engine = pyttsx3.init()

# SETTING UP OF CAMERA TO 1 YOU
# CAN EVEN CHOOSE 0 IN PLACE OF 1
cap = cv2.VideoCapture(0)

# MAIN LOOP IT WILL RUN ALL THE UNLESS
# AND UNTIL THE PROGRAM IS BEING KILLED
# BY THE USER
while True:
	null, frame = cap.read()
	gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
	key = cv2.waitKey(9)
	if key == 27:
		break
cap.release()
cv2.destroyAllWindows()
'''
'''
import cv2
import lib
import pyttsx3
from scipy.spatial import distance

# INITIALIZING THE pyttsx3 SO THAT
# ALERT AUDIO MESSAGE CAN BE DELIVERED
engine = pyttsx3.init()

# SETTING UP OF CAMERA TO 1 YOU CAN
# EVEN CHOOSE 0 IN PLACE OF 1
cap = cv2.VideoCapture(1)

# FACE DETECTION OR MAPPING THE FACE
# TO GET THE Eye AND EYES DETECTED
face_detector = lib.get_frontal_face_detector()

# PUT THE LOCATION OF .DAT FILE (FILE
# FOR PREDECTING THE LANDMARKS ON FACE )
dlib_facelandmark = lib.shape_predictor(
	"C:\\Users\Acer\\Desktop\\geeks\\article 9\\drowsinessDetector-master\\shape_predictor_68_face_landmarks1.dat")

# FUNCTION CALCULATING THE ASPECT RATIO
# FOR THE Eye BY USING EUCLIDEAN DISTANCE
# FUNCTION
def Detect_Eye(eye):
	poi_A = distance.euclidean(eye[1], eye[5])
	poi_B = distance.euclidean(eye[2], eye[4])
	poi_C = distance.euclidean(eye[0], eye[3])
	aspect_ratio_Eye = (poi_A+poi_B)/(2*poi_C)
	return aspect_ratio_Eye


# MAIN LOOP IT WILL RUN ALL THE UNLESS AND
# UNTIL THE PROGRAM IS BEING KILLED BY THE
# USER
while True:
	null, frame = cap.read()
	gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_detector(gray_scale)

	for face in faces:
		face_landmarks = dlib_facelandmark(gray_scale, face)
		leftEye = []
		rightEye = []

		# THESE ARE THE POINTS ALLOCATION FOR THE
		# LEFT EYES IN .DAT FILE THAT ARE FROM 42 TO 47
		for n in range(42, 48):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			rightEye.append((x, y))
			next_point = n+1
			if n == 47:
				next_point = 42
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

		# THESE ARE THE POINTS ALLOCATION FOR THE
		# RIGHT EYES IN .DAT FILE THAT ARE FROM 36 TO 41
		for n in range(36, 42):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			leftEye.append((x, y))
			next_point = n+1
			if n == 41:
				next_point = 36
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

	cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
	key = cv2.waitKey(9)
	if key == 27:
		break
# MAIN LOOP IT WILL RUN ALL THE UNLESS AND
# UNTIL THE PROGRAM IS BEING KILLED BY
# THE USER
while True:
	null, frame = cap.read()
	gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_detector(gray_scale)

	for face in faces:
		face_landmarks = dlib_facelandmark(gray_scale, face)
		leftEye = []
		rightEye = []

		# THESE ARE THE POINTS ALLOCATION FOR
		# THE LEFT EYE IN .DAT FILE THAT ARE
		# FROM 42 TO 47
		for n in range(42, 48):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			rightEye.append((x, y))
			next_point = n+1
			if n == 47:
				next_point = 42
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			# LINE DRAW IN LEFT EYE
			cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

		# THESE ARE THE POINTS ALLOCATION FOR THE
		# RIGHT EYE IN .DAT FILE THAT ARE FROM 36 TO 41
		for n in range(36, 42):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			leftEye.append((x, y))
			next_point = n+1
			if n == 41:
				next_point = 36
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			# LINE DRAW IN RIGHT EYE
			cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

		# CALCULATING THE ASPECT RATIO FOR LEFT
		# AND RIGHT EYE
		right_Eye = Detect_Eye(rightEye)
		left_Eye = Detect_Eye(leftEye)
		Eye_Rat = (left_Eye+right_Eye)/2

		# NOW ROUND OF THE VALUE OF AVERAGE MEAN
		# OF RIGHT AND LEFT EYES
		Eye_Rat = round(Eye_Rat, 2)

		# THIS VALUE OF 0.25 (YOU CAN EVEN CHANGE IT)
		# WILL DECIDE WHETHER THE PERSONS'S EYES ARE
		# CLOSE OR NOT
		if Eye_Rat < 0.25:
			cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
						cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
			cv2.putText(frame, "Alert!!!! WAKE UP DUDE", (50, 450),
						cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)

			# CALLING THE AUDIO FUNCTION OF TEXT TO AUDIO
			# FOR ALERTING THE PERSON
			engine.say("Alert!!!! WAKE UP DUDE")
			engine.runAndWait()
			
	cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
	key = cv2.waitKey(9)
	if key == 20:
		break
cap.release()
cv2.destroyAllWindows()
'''

# install and import above modules first
import os
import cv2
import math
import matplotlib.pyplot as pl
import pandas as pd
from PIL import Image
import numpy as np

# Detect face
def face_detection(img):
	faces = face_detector.detectMultiScale(img, 1.1, 4)
	if (len(faces) <= 0):
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return img, img_gray
	else:
		X, Y, W, H = faces[0]
		img = img[int(Y):int(Y+H), int(X):int(X+W)]
		return img, cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)


def trignometry_for_distance(a, b):
	return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) +\
					((b[1] - a[1]) * (b[1] - a[1])))

# Find eyes
def Face_Alignment(img_path):
	pl.imshow(cv2.imread(img_path)[:, :, ::-1])
	pl.show()
	img_raw = cv2.imread(img_path).copy()
	img, gray_img = face_detection(cv2.imread(img_path))
	eyes = eye_detector.detectMultiScale(gray_img)

	# for multiple people in an image find the largest
	# pair of eyes
	if len(eyes) >= 2:
		eye = eyes[:, 2]
		container1 = []
		for i in range(0, len(eye)):
			container = (eye[i], i)
			container1.append(container)
		df = pd.DataFrame(container1, columns=[
						"length", "idx"]).sort_values(by=['length'])
		eyes = eyes[df.idx.values[0:2]]

		# deciding to choose left and right eye
		eye_1 = eyes[0]
		eye_2 = eyes[1]
		if eye_1[0] > eye_2[0]:
			left_eye = eye_2
			right_eye = eye_1
		else:
			left_eye = eye_1
			right_eye = eye_2

		# center of eyes
		# center of right eye
		right_eye_center = (
			int(right_eye[0] + (right_eye[2]/2)),
		int(right_eye[1] + (right_eye[3]/2)))
		right_eye_x = right_eye_center[0]
		right_eye_y = right_eye_center[1]
		cv2.circle(img, right_eye_center, 2, (255, 0, 0), 3)

		# center of left eye
		left_eye_center = (
			int(left_eye[0] + (left_eye[2] / 2)),
		int(left_eye[1] + (left_eye[3] / 2)))
		left_eye_x = left_eye_center[0]
		left_eye_y = left_eye_center[1]
		cv2.circle(img, left_eye_center, 2, (255, 0, 0), 3)

		# finding rotation direction
		if left_eye_y > right_eye_y:
			print("Rotate image to clock direction")
			point_3rd = (right_eye_x, left_eye_y)
			direction = -1 # rotate image direction to clock
		else:
			print("Rotate to inverse clock direction")
			point_3rd = (left_eye_x, right_eye_y)
			direction = 1 # rotate inverse direction of clock

		cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)
		a = trignometry_for_distance(left_eye_center,
									point_3rd)
		b = trignometry_for_distance(right_eye_center,
									point_3rd)
		c = trignometry_for_distance(right_eye_center,
									left_eye_center)
		cos_a = (b*b + c*c - a*a)/(2*b*c)
		angle = (np.arccos(cos_a) * 180) / math.pi

		if direction == -1:
			angle = 90 - angle
		else:
			angle = -(90-angle)

		# rotate image
		new_img = Image.fromarray(img_raw)
		new_img = np.array(new_img.rotate(direction * angle))

	return new_img


opencv_home = cv2.__file__
folders = opencv_home.split(os.path.sep)[0:-1]
path = folders[0]
for folder in folders[1:]:
	path = path + "/" + folder
path_for_face = path+"/data/haarcascade_frontalface_default.xml"
path_for_eyes = path+"/data/haarcascade_eye.xml"
path_for_nose = path+"/data/haarcascade_mcs_nose.xml"

if os.path.isfile(path_for_face) != True:
	raise ValueError(
		"opencv is not installed pls install using pip install opencv ",
	detector_path, " violated.")

face_detector = cv2.CascadeClassifier(path_for_face)
eye_detector = cv2.CascadeClassifier(path_for_eyes)
nose_detector = cv2.CascadeClassifier(path_for_nose)

# Name of the image for face alignment if on
# the other folder kindly paste the name of
# the image with path included
test_set = ["student.jpeg"]
for i in test_set:
	alignedFace = Face_Alignment(i)
	pl.imshow(alignedFace[:, :, ::-1])
	pl.show()
	img, gray_img = face_detection(alignedFace)
	pl.imshow(img[:, :, ::-1])
	pl.show()
