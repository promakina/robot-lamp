"""
MeVirtuoso.com

Robot Lamp

"""
import os
import logging
from logging.handlers import RotatingFileHandler
import cv2
import serial
import hand_event_detection as ht
import LED_Class
import Robot_Class
import RobotState_Class
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
from ui import RobotLampUI  # Import the UI class

from config_loader import load_parameters
rp = load_parameters("robot_parameters.py")

video_capture = None
running = False
ui = None # ui instance

# Clear log file at startup
log_filename = 'log.log'
if os.path.exists(log_filename):
    os.remove(log_filename)

# Configure RotatingFileHandler
handler = RotatingFileHandler(
    filename=log_filename,
    mode='a',                  # append after initial clear
    maxBytes=1 * 1024 * 1024,  # rotate after reaching 1 MB
    backupCount=1              # retain 1 backup
)

logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def on_rest():
    logging.info('Driving to Rest')
    stop_video()
    LED.Standby()
    robot.driveToRest()
    robot_state.Resting()


def on_closing():
    logging.info('UI_Closed')
    # Stop the video capture if it's running
    stop_video()
    # Set LED
    LED.Standby()
    # Drive Robot
    robot.driveToRest()
    # Close serial
    ser.rts = False
    ser.dtr = False
    time.sleep(0.5)
    ser.close()
    # Destroy the main window to exit the program
    root.destroy()

def stop_video():
    global video_capture, running
    running = False
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    if ui:
        ui.update_video_image(None)  # Clear the video feed

def start_video():
    global video_capture, running
    if not running:
        video_capture = cv2.VideoCapture(rp.CAMERA_INDEX)
        if not video_capture.isOpened():
            logging.error("Error: could not open camera.")
            return
        
        robot.driveHome()
        
        # Adjust camera resolution if needed
        if rp.ADJUST_CAMERA_RESOLUTION:
            logging.debug('Adjust camera resolution')
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, rp.camera_resolution_width)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, rp.camera_resolution_height)
        
        running = True
        
        main_loop()

def main_loop():
    # Initialize offset values
    offset_x = None
    offset_y = None
    
    if running:  
        if ser.in_waiting > 0:
            try: 
                line_raw = ser.readline()
                line = line_raw.decode('utf-8').rstrip()  # Read the response
                logging.debug(f'Message from ESP32: {line}')
            except:
                logging.error(f'Error reading from ESP32, received: {line_raw}')

        # Vision
        result, video_frame = video_capture.read()  # read frames from the video
        
        if not result: return  # terminate the loop if the frame is not read successfully

        # Apply hand tracking and get the offset values.
        flippedFrame, offset_x, offset_y, INDEX_FINGER, TWO_FINGERS, THREE_FINGERS, SHAKA_DETECTED = ( 
            ht.detect_hand(video_frame)  # apply the function we created to the video frame
        )

        robot_state.evaluate(INDEX_FINGER, TWO_FINGERS, THREE_FINGERS, SHAKA_DETECTED)

        cv2_image = cv2.cvtColor(flippedFrame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_image)
        imgtk = ImageTk.PhotoImage(image=pil_image)
        ui.update_video_image(imgtk)

        # Update LED based on state
        if robot_state.current_state.name == 'STANDBY':
            LED.STAY()
        elif robot_state.current_state.name == 'ALERT':
            if robot_state.prev_state.name == 'ACTIVE':
                LED.State1_Tracking()
            else:
                LED.State1_NotTracking()
        elif robot_state.current_state.name == 'ACTIVE':
            LED.Tracking_Enabled()
        elif robot_state.current_state.name == 'HOLD':
            LED.STAY()

        if robot_state.current_state.name == 'STANDBY':
            robot.driveHome(t = 1000)

        # Track
        if robot_state.current_state.name == 'ACTIVE' or robot_state.current_state.name == 'APPROACH':
            robot.find_angles(offset_x, offset_y)
            if robot_state.current_state.name == 'APPROACH': 
                zoom_type = 'IN'
                robot.apply_zoom(zoom_type)
        
            robot.driveRobot()      
        
        root.after(10, main_loop)

if __name__ == "__main__":
    logging.info('Program STARTED')
    
    # Initialize Serial Port
    ser = serial.Serial( rp.COM_PORT, 115200, timeout = 0.2, rtscts=False, dsrdtr=False)  
    time.sleep(0.2)
    # Initialize Robot and LED class
    robot = Robot_Class.Robot(ser)
    LED = LED_Class.LED(ser)
    LED.STAY()
    robot.driveHome(t=2000)

    robot_state = RobotState_Class.RobotStateMachine()
    
    ######################################
    # UI
    ######################################
    root = tk.Tk()

     # Instantiate the UI, passing in the callback functions
    ui = RobotLampUI(
        root,
        on_start=start_video,
        on_stop=stop_video,
        on_close=on_closing,
        on_rest=on_rest,
    )

    root.mainloop()
