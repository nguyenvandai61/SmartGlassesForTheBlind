import RPi.GPIO as IO

buttonPin = 23 #GPIO 23 PIN 16
def initButton():
	global buttonPin
	IO.setwarnings(False)       # do not show any warnings
	IO.setmode (IO.BCM)         # programming the GPIO by BCM pin numbers. (like PIN29 as GPIO5)
	
	IO.setup(buttonPin, IO.IN, pull_up_down=IO.PUD_UP)
								# init button
def isPressed():
	return not IO.input(buttonPin)
