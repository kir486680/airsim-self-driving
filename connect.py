# ready to run example: PythonClient/car/hello_car.py
import airsim
import time
import numpy as np
import cv2
import keras
from keras.models import model_from_json
from keras.models import load_model
from keras.models import Sequential
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
import os
model = load_model('airsim_model.h5')
print("Loaded model from disk")
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img




# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
def get_img():

    # get state of the car
    car_state = client.getCarState()
    #print(car_state)
    #print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

    # set the controls for car
    #car_controls.throttle = 1
    #car_controls.steering = 0
    #client.setCarControls(car_controls)

    # let car drive a bit
    time.sleep(1)

    # get camera images Let me tell you, i setup a software on the adult vids (porn material) website and guess what, you visited this website to experience fun (you know what i mean). When you were watching videos, your internet browser initiated functioning as a Remote Desktop that has a keylogger which gave me access to your display as well as cam. Just after that, my software obtained your entire contacts from your Messenger, Facebook, as well as emailaccount. and then i made a double-screen video. First part shows the video you were viewing (you've got a fine taste lol . . .), and 2nd part shows the view of your webcam, and it is u. from the car
   
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]

    # get numpy array
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 

    # reshape array to 4 channel image array H X W X 4
    img_rgba = img1d.reshape(response.height, response.width, 4)  

    # original image is fliped vertically
    img_rgba = np.flipud(img_rgba)

    # just for fun add little bit of green in all pixels
    #img_rgba[:,:,1:2] = 100

    # write to png 
    airsim.write_png(os.path.normpath("pic.png"), img_rgba) 
    img = cv2.imread("pic.png")
    image = img_preprocess(img)
    image = np.array([image])
    #model = build_model()

    steering_angle = float(model.predict(image))
    print(steering_angle)

    car_controls.throttle = 1
    car_controls.steering = steering_angle
    client.setCarControls(car_controls)
    return img
    #return img1d
        # do something with images
for i in range(100):
    get_img()
#img = get_img()
#cv2.imshow("im" , img)
#cv2.waitKey(0)