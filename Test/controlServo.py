
import RPi.GPIO as IO         
import time   

IO.setwarnings(False)       # do not show any warnings
IO.setmode (IO.BCM)         # numbers
IO.setup(17,IO.OUT)         # initialize GPIO19 as an output, pin #11
IO.setup(22,IO.OUT)         # initialize GPIO19 as an output, pin #15
p1 = IO.PWM(17,50)          # GPIO18 as PWM output, with 50Hz frequency servo 1
p2 = IO.PWM(22,50)          # GPIO19 as PWM output, with 50Hz frequency servo 2
p1.start(2)                 # init
p2.start(2)                 # init 


def servo_quay_trai():
    p1.ChangeDutyCycle(7.5)  
    p2.ChangeDutyCycle(2.5)  #Quay 90 
    time.sleep(1)            # sleep for 1 second

def servo_tro_lai():
    p1.ChangeDutyCycle(2.5)  
    p2.ChangeDutyCycle(2.5)  #quay 90
    time.sleep(1)

def servo_quay_phai():
    p1.ChangeDutyCycle(2.5)  
    p2.ChangeDutyCycle(7.5)  #quay 90 
    time.sleep(1)
    

def servo_cung_quay():
    p1.ChangeDutyCycle(7.5)  #tro lai vi tri ban dau
    p2.ChangeDutyCycle(7.5)  #quay 90 do
    time.sleep(1)
# servo_tro_lai()
# time.sleep(1)
# servo_quay_trai()
# time.sleep(1)
# servo_tro_lai()
# time.sleep(1)
# servo_quay_phai()
while True:
    #value='1'
    value = int(input('Nhap 1 de servo quay va 0 la tro lai ban dau '))
    if(value == 1):
        servo_quay_trai()
        print("quay trai")
    if(value ==2):
        servo_quay_phai()
    if(value ==0):
        servo_cung_quay()
        servo_tro_lai()
        
        