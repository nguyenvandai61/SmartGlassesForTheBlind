import serial
controlData = serial.Serial('COM4',9600)
#sửa lại nếu cổng kết nói board arduino khác com4

def servo_quay_trai():
    controlData.write(str.encode('1'))

def servo_quay_phai():
    controlData.write(str.encode('2'))

def servo_tro_lai():
    controlData.write(str.encode('0'))

while True:
    value = input('nhap 1 de servo quay và 0 là tro lai ban dau ')
    if(value =='1'):
        servo_quay_trai()
    if(value =='2'):
        servo_quay_phai()
    if(value =='0'):
        servo_tro_lai()
   