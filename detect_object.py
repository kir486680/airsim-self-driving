import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as np
import sys
from keras.models import load_model
from dist import process_image
#import import_ipynb
#import book
from connect import get_img
sys.path.append('car_detect/')
options = {
	'model': 'cfg/yolo.cfg',
	'load': 'bin/yolo.weights',
	'threshold': 0.5,
	'gpu': 1.0
}

tfnet = TFNet(options)
#def detect_collision():

def detect(img):
	new_img = np.array(img)

	new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	print(new_img.shape)
	s_img = np.zeros((720, 1080, 3), np.uint8) 
	# use YOLO to predict the image
	result = tfnet.return_predict(img)
	print(len(result))
	print(result)
	print(type(result))
	lines = process_image(img)
	try:
		for i in range(len(result)):
		    
			img_1 = 1 
			img_2 = 1
			#if float(i[4]) > 0.5:
			tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
			br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])
			label = result[i]['label']
			if result[i]['label'] == 'car' and result[i]['confidence'] > 0.5:
				apx_dist = result[i]
		# add the box and label and displaya it
				if tl[0] < 400 : 
					img_1 = cv2.line(s_img , (400 , 1000), br , (0, 0, 255), 2)
					img_2 = cv2.line(new_img , (400 , 1000), br , (0, 0, 255), 2)
				else: 
					img_2 = cv2.line(new_img , (400 , 1000), tl , (0, 0, 255), 2)
					img_1 = cv2.line(s_img , (400 , 1000), tl , (0, 0, 255), 2)
				img_1 = cv2.rectangle(s_img, tl, br, (0, 255, 0), 7)
				img_1 = cv2.putText(s_img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
				
				#img_1 = cv2.line(s_img , (400 , 1000), tl , (0, 0, 255), 2)
				img_2 = cv2.rectangle(new_img, tl, br, (0, 255, 0), 7)
				img_2 = cv2.putText(new_img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
				
				#img_2 = cv2.line(new_img , (400 , 1000), tl , (0, 0, 255), 2)
		for line in lines:
			for x1,y1,x2,y2 in line:
				img_2 = cv2.line(new_img,(x1,y1),(x2,y2),(255,0,0),10)
				img_1 = cv2.line(s_img,(x1,y1),(x2,y2),(255,0,0),10)
		return img_1 , img_2
	except IndexError:
		return img
'''
img_1 , img_2 = detect(cv2.imread("pic.png"))
cv2.imshow("im" , img_1)
cv2.imshow("img" , img_2)
cv2.waitKey(20000)
'''
while 1:
	img = get_img()
	print(type(img))
	#img = np.array(img)
	print(img.shape)

	#img = cv2.imread(img)
	img_1 = 2
	img_2 = 1
	img_1 , img_2 = detect(img)
	cv2.imshow("im" , img_1)
	if cv2.waitKey(0) & 0xFF == ord('q'):
		break
