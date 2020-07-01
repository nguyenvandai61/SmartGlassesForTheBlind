# My modules
from utils.SoundPlayer import playSound
from models import DetectWay, DetectObject
from servos import Servos
from button import Button
# processing-image image

import imutils
import numpy as np
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
from picamera import PiCamera, Color

dirr = "Downloads/DA/"
models_dirr = dirr + "models/"
sounds_dirr = dirr + "sounds/"
video_dirr = dirr + "video/"
MobileNetSSDDir = models_dirr+ "MobileNetSSD/"

weight_file = models_dirr + "model.hdf5"
video_output_file = video_dirr+"test.h264"
model = MobileNetSSDDir+"MobileNetSSD_deploy.caffemodel"
prototxt = MobileNetSSDDir +"MobileNetSSD_deploy.prototxt.txt"

directions=['center', 'left', 'right']
labels_name={'center':0,'left':1,'right':2}
# Servos and button
Servos.initServos()
Button.initButton()
# Model
DetectWay.initModel()
DetectWay.loadWeight(weight_file)
DetectObject.loadModel(prototxt, model)


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (720, 480)
camera.framerate = 30

idxs = [0, 0, 0]
print("Thu bam nut")
    
while True:
    # Neu button dang nhan
    if Button.isPressed():
    
        print("Nut dang duoc bam")
        rawCapture = PiRGBArray(camera, size=(720, 480))
        camera.start_preview()     
        camera.annotate_background = Color("blue")
        camera.start_recording(video_output_file)
        i = 0
        for frame in camera.capture_continuous(rawCapture, format= "bgr", use_video_port = True):
            print(i)
            if (i%15 == 0):
                test_image = rawCapture.array
                test_image2 = test_image;
                test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
   
                print ("Xu ly anh")
                #test_image = cv2.imread('left.jpg', 0);
                #print(test_image.shape)
                #print(test_image.size)
                test_image=cv2.resize(test_image,(128,128))
                #cv2.imshow('asdggg', test_image)
                #cv2.waitKey(2)
                test_image = np.array(test_image)
                test_image = test_image.astype('float32')
                test_image /= 255

                test_image = test_image.reshape(1,128,128,1)
                value = DetectWay.predict_class(test_image)
                direction = directions[value];
                 
                print("Day la: "+direction)
                camera.annotate_text = direction + " "+  str(DetectWay.predict(test_image)[value])
                soundfile = sounds_dirr + direction + '.mp3'
                playSound(soundfile)
               
# #                         filename = dirr+"newdata/"+direction+"/"+str(idxs[value])+".jpg"
# #                         idxs[value]+=1
# #                         print("ABC")
# #                         cv2.imwrite(filename,saved_image) #save image
# #                         print("Luu anh")
                frame = imutils.resize(test_image2, width=300)
                detect = DetectObject.process(frame)
                DetectObject.predict(detect, 0.2)
                # DetectObject.drawOnImage(camera, detect)
                Servos.rotate(value)
                
                
            i = i + 1
            if i==300:   
                camera.stop_recording()
                camera.stop_preview()
                exit()
            rawCapture.truncate(0)
        
