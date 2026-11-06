#include <Servo.h>
#include <stdlib.h>

const int SERVO_PIN = 6;
const int MIN_ANGLE = 20;
const int MAX_ANGLE = 160;

Servo panServo;
int targetAngle = 90;
char serialBuffer[16];
byte serialIndex = 0;

void setTargetAngle(int value)
{
  if (value < MIN_ANGLE)
  {
    value = MIN_ANGLE;
  }
  else if (value > MAX_ANGLE)
  {
    value = MAX_ANGLE;
  }
  targetAngle = value;
  panServo.write(targetAngle);
}

void setup()
{
  Serial.begin(115200);
  panServo.attach(SERVO_PIN);
  panServo.write(90);
  targetAngle = 90;
}

void loop()
{
  while (Serial.available() > 0)
  {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r')
    {
      if (serialIndex > 0)
      {
        serialBuffer[serialIndex] = '\0';
        bool valid = true;
        for (byte i = 0; i < serialIndex; i++)
        {
          char digit = serialBuffer[i];
          if (i == 0 && (digit == '-' || digit == '+'))
          {
            continue;
          }
          if (digit < '0' || digit > '9')
          {
            valid = false;
            break;
          }
        }
        if (valid)
        {
          setTargetAngle(atoi(serialBuffer));
        }
        serialIndex = 0;
      }
    }
    else if (serialIndex < sizeof(serialBuffer) - 1)
    {
      serialBuffer[serialIndex++] = ch;
    }
    else
    {
      serialIndex = 0;
    }
  }

}
