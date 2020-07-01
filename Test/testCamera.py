from tensorflow.python.keras.models import load_model
#import os
import cv2
import numpy as np 
#import matplotlib.pyplot as plt 
#from cv2 import *
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten 
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from picamera import PiCamera

# initialize the camera
camera = PiCamera()
image = camera.capture('/home/pi/Desktop/a.jpg')
cv2.imshow('dd', image);

waitKey(1500)  
destroyWindow("cam-test");
waitKey(1500)
