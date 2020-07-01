import RPi.GPIO as IO 
import time

p1 = None;
p2 = None;
def initServos():
	global p1, p2 
	# Controll Servo
	IO.setwarnings(False)       # do not show any warnings
	IO.setmode (IO.BCM)         # programming the GPIO by BCM pin numbers. (like PIN29 as GPIO5)
	IO.setup(17,IO.OUT)         # initialize GPIO19 as an output, pin #11
	IO.setup(22,IO.OUT)         # initialize GPIO19 as an output, pin #15
	p1 = IO.PWM(17,50)          # GPIO18 as PWM output, with 50Hz frequency servo 1
	p2 = IO.PWM(22,50)          # GPIO19 as PWM output, with 50Hz frequency servo 2
	p1.start(2)                 # init 
	p2.start(2)                 # init (bat dau tai chu ki xung 2)


def quay_trai():
	global p1, p2
	p1.ChangeDutyCycle(7.5)
    # Quay 90 do
	p2.ChangeDutyCycle(2.5)  # Quay 90 do
	time.sleep(1)            # sleep for 1 second

def tro_lai():
	global p1, p2
	p1.ChangeDutyCycle(2.5)  #tro lai vi tri ban dau
	p2.ChangeDutyCycle(2.5)  #quay 90 do
	time.sleep(1)

def quay_phai():
	global p1, p2
	p1.ChangeDutyCycle(2.5)  #tro lai vi tri ban dau
	p2.ChangeDutyCycle(7.5)  #quay 90 do
	time.sleep(1)
    
def cung_quay():
	global p1, p2
	p1.ChangeDutyCycle(7.5)  #tro lai vi tri ban dau
	p2.ChangeDutyCycle(7.5)  #quay 90 do
	time.sleep(1)
	
def rotate(value):
	if (value == 0):
		cung_quay()
		tro_lai()
	if (value == 1):
		quay_trai();
		tro_lai();
	if (value == 2):
		quay_phai()
		tro_lai()
	

