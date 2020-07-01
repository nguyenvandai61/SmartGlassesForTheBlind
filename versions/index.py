from tensorflow.python.keras.models import load_model
import numpy as np
from playsound import playsound
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten 
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D 
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import RPi.GPIO as IO 

# Controll Servo
IO.setwarnings(False)       # do not show any warnings
IO.setmode (IO.BCM)         # programming the GPIO by BCM pin numbers. (like PIN29 as GPIO5)
IO.setup(17,IO.OUT)         # initialize GPIO19 as an output, pin #11
IO.setup(22,IO.OUT)         # initialize GPIO19 as an output, pin #15
p1 = IO.PWM(17,50)          # GPIO18 as PWM output, with 50Hz frequency servo 1
p2 = IO.PWM(22,50)          # GPIO19 as PWM output, with 50Hz frequency servo 2
p1.start(2)                 # init
p2.start(2)                 # init (bat dau tai chu ki xung 2)

buttonPin = 23 #GPIO 23 PIN 16
IO.setup(buttonPin, IO.IN, pull_up_down=IO.PUD_UP)
                            # init button

def servo_quay_trai():
    #p.ChangeDutyCycle(d)
    p1.ChangeDutyCycle(7.5)  #Quay 90 do
    p2.ChangeDutyCycle(2.5)  #Quay 90 do
    time.sleep(1)            # sleep for 1 second

def servo_tro_lai():
    #p.ChangeDutyCycle(d)
    p1.ChangeDutyCycle(2.5)  #tro lai vi tri ban dau
    p2.ChangeDutyCycle(2.5)  #quay 90 do
    time.sleep(1)

def servo_quay_phai():
    p1.ChangeDutyCycle(2.5)  #tro lai vi tri ban dau
    p2.ChangeDutyCycle(7.5)  #quay 90 do
    time.sleep(1)
def servo_cung_quay():
    p1.ChangeDutyCycle(7.5)  #tro lai vi tri ban dau
    p2.ChangeDutyCycle(7.5)  #quay 90 do
    time.sleep(1)
img_rows=128
img_cols=128
num_channel=1
num_epoch=20

# Define the number of classes
num_classes = 3
labels_name={'center':0,'left':1,'right':2}

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128,128,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5)) 
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# Viewing model_configuration

# model.summary()
# model.get_config()
# model.layers[0].get_config()
# model.layers[0].input_shape         
# model.layers[0].output_shape            
# model.layers[0].get_weights()
# np.shape(model.layers[0].get_weights()[0])
# model.layers[0].trainable
# load model
dirr = "Downloads/DA/"
model.load_weights(dirr+'model.hdf5')
#model.load_model('model.hdf5')
#labels_name={'center':0,'left':1,'right':2}
directions={'center', 'left', 'right'}

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (720, 480)
idx = [0, 0, 0]

print("Thu bam nut")
    
while True:
    buttonState = IO.input(buttonPin)
    # Neu button dang nhan
    if buttonState == False:
        print("Nut dang duoc bam")
        rawCapture = PiRGBArray(camera, size=(720, 480))
        # allow the camera to warmup
        #time.sleep()
        
        # grab an image from the camera
        camera.capture(rawCapture, format="bgr")
        test_image = rawCapture.array
        saved_image = test_image
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        print(test_image.shape)
        print(test_image.size)

        #cv2.imshow('asd', test_image)
        #cv2.waitKey(2)
        #cv2.destroyWindow('asd')
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
        #val = model.predict(test_image)
        #print(val)
        #print(model.predict_classes(test_image))
        value = model.predict_classes(test_image)[0];
        print(value)
        
        
        for direction in directions:
                  #print(dataset)
            if labels_name[direction]==model.predict_classes(test_image)[0]:
                print("Day la:"+direction)
                            #playsound('./mp3/'+ dataset + '.mp3')
                filename = dirr+"newdata/"+direction+"/"+str(idx[value])+".jpg"
                idx[value]+=1
                print("ABC")
                cv2.imwrite(filename,saved_image) #save image
                print("Luu anh")
                
                break
        if (value == 0):
            servo_quay_trai()
            servo_quay_phai()
            #print("dasdfs")
            servo_tro_lai()
        if (value == 1):
            servo_quay_trai();
            servo_tro_lai();
             #       break
        if (value == 2):
            servo_quay_phai()
            servo_tro_lai()
        #cv2.imshow('test',test_image.reshape(128,128))    
        #cv2.waitKey(1000)  
        #cv2.destroyAllWindows();
        #cv2.waitKey(1500)
        time.sleep(2)
    if idx[0]+idx[1]+idx[2] >= 5:
        break