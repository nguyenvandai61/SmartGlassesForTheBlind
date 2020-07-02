# Cac modules
from utils.SoundPlayer import playSound
from models import DetectWay, DetectObject
from servos import Servos
from button import Button

import io
import socket
import time
import imutils
import numpy as np
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
from picamera import PiCamera, Color

def save_image(image, value):
        global idxs
        filename = dirr+"newdata/"+direction+"/"+str(idxs[value])+".jpg"
        idxs[value]+=1
        cv2.imwrite(filename, image)
dirr = "Downloads/DA/"
models_dirr = dirr + "models/"
sounds_dirr = dirr + "sounds/"
video_dirr = dirr + "video/"
MobileNetSSDDir = models_dirr+ "MobileNetSSD/"

weight_file = models_dirr + "model07012020.h5"
#weight_file = models_dirr + "model.hdf5"
video_output_file = video_dirr+"test.h264"
model = MobileNetSSDDir+"MobileNetSSD_deploy.caffemodel"
prototxt = MobileNetSSDDir +"MobileNetSSD_deploy.prototxt.txt"

directions=['center', 'left', 'right']
labels_name={'center':0,'left':1,'right':2}
# Khoi tao servos va button
Servos.initServos()
Button.initButton()
# Model
DetectWay.initModel()
DetectWay.loadWeight(weight_file)
DetectObject.loadModel(prototxt, model)


camera = PiCamera()
camera.resolution = (720, 480)
camera.framerate = 30

idxs = [0, 0, 0]
print("Press Button!!")
    
while True:
    # Neu button dang nhan
    if Button.isPressed():
        print("Button is pressed")
        rawCapture = PiRGBArray(camera, size=(720, 480))
        camera.start_preview()     
        camera.annotate_background = Color("blue")
        camera.start_recording(video_output_file)
        i = 0
        for frame in camera.capture_continuous(rawCapture, format= "bgr", use_video_port = True):
            print(i)
            # 15 frame lay 1 frame xu ly
            if (i%10 == 0):
                test_image = rawCapture.array
                
                # Xu ly anh
                start_time1 = time.time()
                test_image2 = test_image;
                
                
                test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                test_image=cv2.resize(test_image,(128,128))
                
                
                
                
                test_image = np.array(test_image)
                test_image = test_image.astype('float32')
                test_image /= 255

                test_image = test_image.reshape(1,128,128,1)
                # Du doan duong di
                value = DetectWay.predict_class(test_image)
                direction = directions[value];
                print("This is: "+direction)
                finish_time1 = time.time()
                print("Total detectway-time: %f s", finish_time1 - start_time1)                
                
                # Hien thi text tren video
                camera.annotate_text = direction + " "+  str(DetectWay.predict(test_image)[value])
                # Quay servos
                Servos.rotate(value)
                # Phat qua tai nghe
                soundfile = sounds_dirr + direction + '.mp3'
                playSound(soundfile)

                # Xu ly nhan dien vat the
                start_time2 = time.time()
                frame = imutils.resize(test_image2, width=300)
                detect = DetectObject.process(frame)
                # Cho ket qua du doan
                confidence = 0.8
                DetectObject.predict(detect, confidence)
                DetectObject.drawCamera(camera, detect, confidence)
                finish_time2 = time.time()
                print("Total detectobj-time: %f s", finish_time2 - start_time2)                
                

                # Luu anh
                save_image(test_image2, value)
                
                process_time = finish_time2 - start_time1
                i += int(process_time*30)
                
                
            i = i + 1
            if i>1500:
                    
                
                camera.stop_recording()
                camera.stop_preview()
                
                server_socket.close()
                exit()
            rawCapture.truncate(0)

