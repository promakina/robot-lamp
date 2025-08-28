# **Robot Lamp \- Hand-Tracking LED Light**

This repository contains all the necessary files, code, and documentation for the Robot Lamp, a hand-tracking robotic lamp that points an LED light in the direction of a user's hand.

The system uses a PC with a webcam to run computer vision analysis and an ESP32 microcontroller to handle the low-level motor control. All mechanical components are designed to be 3D printed.

Product page: [https://mevirtuoso.com/product/robot-lamp/](https://mevirtuoso.com/product/robot-lamp/) 

## **Features**

* **Real-time Hand Tracking:** Utilizes OpenCV and MediaPipe Hands on a PC to detect hand position and gestures.  
* **Gesture Control:** The robot's state can be controlled through hand gestures, allowing for tracking, standby, and zoom modes.  
* **Programmable RGB LED:** Features a customizable RGB LED for lighting.  
* **Serial Bus Servos:** Employs strong and affordable Serial Bus servos for precise movement and clean, simple wiring.  
* **3D Printable:** All structural components are designed to be 3D printed with standard metric hardware.

## **Repository Structure**

* /esp32\_code/: Contains the C++ source code for the ESP32 microcontroller, which controls the motors and LED.  
* /python/: Contains the Python script that runs on the PC for hand tracking and gesture recognition via OpenCV.

## **Hardware Components**

* **Microcontroller:** ESP32  
* **Motors:** Serial Bus Servos  
* **Lighting:** Programmable RGB LED  
* **PC with Webcam:** Required for running the Python-based computer vision software.  
* **3D Printed Parts:** Custom-designed frame and linkages.  
* **Fasteners:** Standard metric hardware.

## **Getting Started**

### **1\. 3D Printing and Assembly**

Begin by printing all the necessary components and assemble the robot according to the provided guide (link to be added). Files and guides can be found on the product page: [https://mevirtuoso.com/product/robot-lamp/](https://mevirtuoso.com/product/robot-lamp/) 

### **2\. ESP32 Setup**

1. Flash the firmware located in the /esp32\_code directory to your ESP32 board.  
2. Wire the motors and LED to the ESP32 as per the schematic (link to be added).

### **3\. PC Setup**

1. Ensure you have Python installed on your PC.  
2. Install the required libraries:  
   pip install opencv-python mediapipe pyserial

3. Connect the ESP32 to your PC via USB.  
4. Run the main Python script from the /pc\_python\_code directory.

## **License**

This project is licensed under the **Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)**. Please see the LICENSE file for full details.

*Copyright © 2025 MeVirtuoso.com*