#include <Arduino.h>
#include <SCServo.h>
#include "LEDControl.h"
#include "motor_parameters.h"

#define BUTTON_PIN 15  // Pin where the switch is connected
boolean LED_ON = true; // controlled by switch, turn on/off LEDs
SCSCL sc;
float ts, ts2;
boolean posA = true;
int t = 0; // Default time to drive servo
int s = 0; // Default servo speed

int j1Angle = 0, j2Angle = 0, j3Angle = 0, j4Angle = 0;
int j1Angle_new = 0, j2Angle_new = 0, j3Angle_new = 0, j4Angle_new = 0;

// Function declaration
bool isNumeric(String str);
bool commaCountCorrect(String data);

// ------------------------------
// Breathing Effect Variables
bool breathingMode = false;
CRGB breathColor;
uint8_t breathBrightness;
// ------------------------------
void setup() 
{
  ts , ts2 = millis();
  Serial.begin(115200); // For serial monitor
  Serial2.begin(1000000); // For servo communication
  sc.pSerial = &Serial2;
  delay(200);
  Serial.println("SCServo Test");

  int t = 1000;
  int s = 0;
  
  motor1.driveToDegree(0,t,s,sc);
  motor2.driveToDegree(0,t,s,sc);
  motor3.driveToDegree(0,t,s,sc);
  motor4.driveToDegree(0,t,s,sc);
  motor5.driveToDegree(0,t,s,sc);
  
  delay(1500);
  
  pinMode(BUTTON_PIN, INPUT_PULLUP);  // Use internal pull-up resistor for ON/OFF button

  // Enable Torque
  for (int i=1;i<=6;i++) 
  {
    sc.EnableTorque(i, 0);
  }
  delay(200);
  
  Serial.println("Motors Centered");
  delay(100);
  setupLEDs();
  turnOnAllLEDs(CRGB::Red, 200);  // Turn on all LEDs to white at brightness 200
  delay(500);
  
}

void loop() {
  if (digitalRead(BUTTON_PIN) == LOW) {  // Button is pressed (LOW because of pull-up)
    Serial.println("Button Pressed!");
    LED_ON = !LED_ON;
    if (!LED_ON) {
      turnOffLEDs();
    }
    delay(300);  // Debounce delay
  } 

  if (!LED_ON) return;

  if (Serial.available() > 0) // Read from Serial
  {
    String data = Serial.readStringUntil('\n');
    data.trim();

    //Serial.print("ESP32 | Data_Received: ");
    //Serial.print(data);    

    if (data.startsWith("LED")) {
      processLEDCommand(data);  // Send the full message to be processed
      
      // If the command was for breathing, enable breathing mode
      if (data.indexOf("BREATH") != -1) {
        breathingMode = true;  // Enable continuous breathing
      } 
      else {
        breathingMode = false; // Disable breathing mode for other commands
      }
      return;  // Exit loop() early, skipping further processing
    }
    
    // ELSE, process the message as a servo command

    // Check Comma numbers
    if (!commaCountCorrect(data)) {
      Serial.println("Incorrect format: Comma Count Incorrect.");
      return; // Check there are 4 commas, if not, skip
    }

    int firstCommaIndex = data.indexOf(',');
    int secondCommaIndex = data.indexOf(',', firstCommaIndex + 1);
    int thirdCommaIndex = data.indexOf(',', secondCommaIndex + 1);
    int fourthCommaIndex = data.indexOf(',', thirdCommaIndex + 1);
    String j1Str = data.substring(0, firstCommaIndex);
    String j2Str = data.substring(firstCommaIndex+1, secondCommaIndex);
    String j3Str = data.substring(secondCommaIndex+1, thirdCommaIndex);
    String j4Str = data.substring(thirdCommaIndex+1, fourthCommaIndex);
    String tStr = data.substring(fourthCommaIndex+1);

    // Check all parts are numeric
    bool formatCorrect =  
      isNumeric(j1Str) && isNumeric(j2Str) &&
      isNumeric(j3Str) && isNumeric(j4Str) &&
      isNumeric(tStr);

    if (!formatCorrect) {
      Serial.println("Incorrect format: Non-numeric values detected.");
      return;
    }
  
    float j1Float = j1Str.toFloat();
    float j2Float = j2Str.toFloat();
    float j3Float = j3Str.toFloat();
    float j4Float = j4Str.toFloat();
    float tFloat = tStr.toFloat();
    j1Angle_new = static_cast<int>(j1Float);
    j2Angle_new = static_cast<int>(j2Float);
    j3Angle_new = static_cast<int>(j3Float);
    j4Angle_new = static_cast<int>(j4Float);
    t = static_cast<int>(tFloat);
    
  }

  // Run the breathing effect continuously if enabled
  if (breathingMode) {
    breathingEffect(breathColor, breathBrightness);
    //Serial.println("Brightnes Enabled");
  }

  if (millis()-ts2 > 10) // Small Delay to Drive servos
  { 

    j1Angle = j1Angle_new;
    j2Angle = j2Angle_new;
    j3Angle = j3Angle_new;
    j4Angle = j4Angle_new;

    motor1.driveToDegree(j1Angle,t,s, sc);
    motor2.driveToDegree(j2Angle,t,s, sc);
    motor3.driveToDegree(j2Angle,t,s, sc);
    motor4.driveToDegree(j3Angle,t,s, sc);
    motor5.driveToDegree(j4Angle,t,s, sc); 
    ts2 = millis();
  }
  
}

bool isNumeric(String str) {
  if (str.length() == 0) return false;

  bool hasDecimal = false;
  int startIdx = 0;

  // Handle optional leading sign (+/-)
  if (str.charAt(0) == '-' || str.charAt(0) == '+') {
      startIdx = 1;
      if (str.length() == 1) return false; // can't be only "+" or "-"
  }

  for (int i = startIdx; i < str.length(); i++) {
      char c = str.charAt(i);
      if (c == '.') {
          if (hasDecimal) return false; // only allow one decimal point
          hasDecimal = true;
      }
      else if (!isDigit(c)) {
          return false; // any non-digit and non-dot character is invalid
      }
  }
  return true;
}

bool commaCountCorrect(String data) {
    // Make sure input data has the correct format (3 commas)
    int commaCount = 0;
    int expectedNumberCommas = 4;
    for (int i = 0; i < data.length(); i++)
    {
      if (data.charAt(i) == ',') commaCount++;
    }

    // Check for exactly four commas
    if (commaCount != expectedNumberCommas) {
      Serial.println("Incorrect format: Wrong number of commas.");
      return false;
    }
    return true;
}
