#include<Servo.h>

char controll;
Servo servo1;
Servo servo2;
void setup() {
  Serial.begin(9600);
  servo1.attach(9); // kết nối chân 9 trên arduino với chân điều khiển của servo
  servo2.attach(10);
  Serial.print("Start");
  servo1.write(90);
  servo2.write(90);
  delay(1000);
  servo1.write(0);
  servo2.write(0); 
}

void loop() {
  if(Serial.available()>0)
  {
    controll = Serial.read();
    Serial.print(controll);
    if(controll=='1'){
      servo1.write(90);//Quay 90 độ
      
      servo2.write(0);//Quay 90 độ
      delay(1000);
    }
    if(controll=='0'){
      servo1.write(0);//trở lại vị trí ban đầu
      servo2.write(0);//Quay 90 độ
      delay(1000);
    }
     if(controll=='2'){
      servo1.write(0);//trở lại vị trí ban đầu
      servo2.write(90);//Quay 90 độ
      delay(1000);
    }
    
  }

}
