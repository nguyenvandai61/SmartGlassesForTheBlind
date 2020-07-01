from tensorflow.python.keras.models import load_model
import numpy as np
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten 
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D 

# Define the number of classes
model = None
num_classes = 3
labels_name={'center':0,'left':1,'right':2}
directions={'center', 'left', 'right'}


def initModel():
	global model
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

def viewModelConfig():
	global model
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
def loadWeight(file_dir):	
	global model
	model.load_weights(file_dir)
	
def predict_class(image):
	global model
	index_label = model.predict_classes(image)[0]
	return index_label
	
def predict(image):
	global model
	acc = model.predict(image)[0];
	return acc
