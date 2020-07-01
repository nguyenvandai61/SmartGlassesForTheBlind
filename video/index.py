from picamera import PiCamera, Color
from picamera.array import PiRGBArray
import cv2
from time import sleep
import datetime as dt

camera = PiCamera()
#camera = PiCamera()
camera.resolution = (720, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(720, 480))
camera.start_preview()     
camera.annotate_background = Color("blue")
camera.start_recording('/home/pi/'+'test.h264')
i = 0
for frame in camera.capture_continuous(rawCapture, format= "bgr", use_video_port = True):
    print(i)
    if (i%30 == 0):
        camera.annotate_text = str(i)
        test_image = rawCapture.array
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('sadad',test_image)
    i = i + 1
    if i==300:   
        camera.stop_recording()
        camera.stop_preview()
        exit()
    rawCapture.truncate(0)  
# def quayvideo(title):
#     camera.start_preview()
#     
#     camera.annotate_background = Color("blue")
#     start = dt.datetime.now()
#     
#     camera.start_recording('/home/pi/'+title+'.h264')
#     
#     while (dt.datetime.now() - start).seconds < 10:
#         camera.annotate_text = dt.datetime.now().strftime('%d-%m-%Y_%Hh%Mm%S')
#         camera.wait_recording(0.2)
# 
#     camera.stop_recording()
#     camera.stop_preview()
# 
# 
# print('Bat dau quay!')
# quayvideo('trinh')
