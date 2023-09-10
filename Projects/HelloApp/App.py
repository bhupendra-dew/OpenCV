from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import webbrowser

class Ui_MainWindow(QtWidgets.QWidget):
	def setupUi(self, MainWindow):
		MainWindow.resize(422, 255)
		self.centralwidget = QtWidgets.QWidget(MainWindow)

		self.pushButton = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton.setGeometry(QtCore.QRect(160, 130, 93, 28))

		# For displaying confirmation message along with user's info.
		self.label = QtWidgets.QLabel(self.centralwidget)
		self.label.setGeometry(QtCore.QRect(170, 40, 201, 111))

		# Keeping the text of label empty initially.	
		self.label.setText("")	

		MainWindow.setCentralWidget(self.centralwidget)
		self.retranslateUi(MainWindow)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)

	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
		self.pushButton.setText(_translate("MainWindow", "Proceed"))
		self.pushButton.clicked.connect(self.takeinputs)
		
	def takeinputs(self):
		IPv4, done = QtWidgets.QInputDialog.getText(
			self, 'Input Dialog', 'Enter your Mobile IPv4 Address : ')

		if done:
			# Showing confirmation message along
			# with information provided by user.
			self.label.setText('Information stored Successfully\nIPv4 Address : '
								+str(IPv4))
			cap = cv2.VideoCapture(str(IPv4))
			detector = cv2.QRCodeDetector()
			while True:
				_, img = cap.read()
				print(img)
				data, bbox, _ = detector.detectAndDecode(img)
				
				if data:
					a = data
					break
				cv2.imshow('QR_CODE_Scanner', img)
				k = cv2.waitKey(1) & 0xFF 
				if k == 27:
					break
			b = webbrowser.open(str(a))
			cap.release()
			cv2.destroyAllWindows()

			# Hide the pushbutton after inputs provided by the use.
			self.pushButton.hide()
				
			
			
			
if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(MainWindow)
	MainWindow.show() 

	sys.exit(app.exec_())
