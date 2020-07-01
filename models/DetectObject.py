import numpy as np
import cv2
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = None
detections = None

def loadModel(prototxt, model):
	global net
	print("Loading model...")
	net = cv2.dnn.readNetFromCaffe(prototxt, model)

def process(frame):
	global net, detections
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                  0.007843, (300, 300), 127.5)
                                  
	net.setInput(blob)
	detections = net.forward()
	return detections

def predict(detections, confidence_threshold):
    for i in np.arange(0, detections.shape[2]):
		# Get the confidence
	    confidence = detections[0, 0, i, 2]
	    if confidence > confidence_threshold:
		value = int(detections[0, 0, i, 1])
		label = "{}: {:.2f}%".format(CLASSES[value], confidence * 100)
		print(label)

def drawOnImage(frame, detections):
    for i in np.arange(0, detections.shape[2]):
	value = int(detections[0, 0, i, 1])
	box = detections[0, 0, i, 3:7] * np.array([1, 1, 1, 1])
	print(box)
	(startX, startY, endX, endY) = box.astype("int")
		    
	cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[value], 2)
	y = startY - 15 if startY - 15 > 15 else startY + 15
	cv2.putText(frame, label, (startX, y),
	    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[value], 2)
		    
