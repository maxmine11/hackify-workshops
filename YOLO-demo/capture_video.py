import cv2 
import numpy as np 
import subprocess
import time

cap = cv2.VideoCapture(0)

while(True):
	##Capture frame by frame
	ret, frame = cap.read() 

	# create image for call for inferance
	cv2.imwrite("data/frame.jpg", frame)

	#run darknet on bash with frame.jpg
	subprocess.call('./darknet detector test cfg/voc.data cfg/tiny-yolo-voc.cfg tiny-yolo-voc.weights data/frame.jpg', shell=True)

	#time wait


	# read predictions.png to display object detection
	prediction = cv2.imread("predictions.png")


	#Display the resulting frame 
	cv2.imshow('Camera_frame', prediction)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# when everything is done we need to realese the capture

cap.release()
cv2.destroyAllWindows()