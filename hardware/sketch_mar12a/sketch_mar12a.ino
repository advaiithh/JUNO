#include <Servo.h>

Servo myServo;

void setup() {
  myServo.attach(9);
  myServo.write(90);   // move to center
}

void loop() {
}