from tensorflow.python.keras.models import load_model
import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from cv2 import *
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten 
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D 
import serial
controlData = serial.Serial('COM4',9600)
#sửa lại nếu cổng kết nói board arduino khác com4


data_path = 'data'
data_dir_list = os.listdir(data_path)
img_rows=128
img_cols=128
num_channel=1
num_epoch=20

# Define the number of classes
num_classes = 3

labels_name={'center':0,'left':1,'right':2}

# Send the code to COM Serial
def servo_quay_trai():
    controlData.write(str.encode('1'))

def servo_quay_phai():
    controlData.write(str.encode('2'))

def servo_tro_lai():
    controlData.write(str.encode('0'))



value = 0
# initialize the camera
cam = VideoCapture(0)   # 0 -> index of camera

# 5. Định nghĩa model 
model = Sequential()
# Thêm Convolutional layer với 32 kernel, kích thước kernel 3*3 
# dùng hàm relu làm activation và chỉ rõ input_shape cho layer đầu tiên 
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128,128,1)))
# Thêm Convolutional layer 1
model.add(Conv2D(32, (3, 3), activation='relu'))
# Thêm Max pooling layer 1
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
# Thêm Convolutional layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
# Thêm Max pooling layer 2
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
# Flatten layer chuyển từ tensor sang vector 
model.add(Flatten())
# Thêm Fully Connected layer với 64 nodes và dùng hàm relu 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# Output layer với 10 node và dùng softmax function để chuyển sang xác suất.
model.add(Dense(num_classes, activation='softmax'))
# 6. Compile model, chỉ rõ hàm loss_function nào được sử dụng, phương thức 
# đùng để tối ưu hàm loss function. 
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# Viewing model_configuration

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable
# load model
model.load_weights('model.hdf5')

# 10. Dự đoán ảnh 
#test_image = cv2.imread('test/03.jfif')
while True:
    s, test_image = cam.read()
    

    if s:    # frame captured without any errors
        #namedWindow("cam-test",CV_WINDOW_AUTOSIZE)
        #imshow("cam-test",test_image)
        #cv2.imshow('test',test_image);
        waitKey(0)
        destroyWindow("cam-test")
        #test_image=cv2.resize(test_image,(128,128))
        #imwrite("test/test.jpg",test_image) #save image
        # waitKey(30);
        print ("Xu ly anh")
        # test_image = cv2.imread('test/test.jpg')
        cv2.imshow('test',test_image);
        test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        test_image=cv2.resize(test_image,(128,128))
        test_image = np.array(test_image)
        test_image = test_image.astype('float32')
        test_image /= 255

        test_image = test_image.reshape(1,128,128,1)

        print(model.predict(test_image))
        print(model.predict_classes(test_image))
        value = model.predict_classes(test_image)[0];
        #cv2.imshow(test_image,1)
        for dataset in data_dir_list:
          #print(dataset)
          if labels_name[dataset]==model.predict_classes(test_image)[0]:
            
            print("Day la:"+dataset)
            break
        if value == 1:
            servo_quay_trai()
        if value == 2:
            servo_quay_phai()
        if value == 0:
            servo_tro_lai()
        
        #plt.imshow(test_image.reshape(128,128), cmap="gray")
        waitKey(0)
        break
        waitKey(3000)
