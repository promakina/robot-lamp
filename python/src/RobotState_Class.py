from config_loader import load_parameters

rp = load_parameters("robot_parameters.py")


import logging
import time
from enum import Enum

'''
States:
    standby = No action needed
    alert = intermediate state, ready to switch to tracking, or go back to standby
    active = track hand, keep it in the FOV
    approach = track hand and also zoom in if possible

'''

class RobotState(Enum):
    STANDBY = "standby"
    ALERT = "alert"
    ACTIVE = "active"
    HOLD = "hold position"
    APPROACH = "approach"

class RobotStateMachine:
    def __init__(self):
        self.prev_state = RobotState.STANDBY
        self.current_state = RobotState.STANDBY

        self.reset_timeout = rp.RESET_TIMEOUT
        self.gesture_duration = rp.FIRST_GESTURE_DURATION
        self.zoom_duration = rp.ZOOM_GESTURE_DURATION
        self.reset_duration = rp.RESET_GESTURE_DURATION
        
        self.first_gesture_timer = None
        self.zoom_gesture_timer = None
        self.reset_timer = None
        self.alert_timer = None

    def transition_to(self, new_state):
        if self.current_state != new_state: 
            logging.debug(f'ROBOT_STATE transition to {new_state}')
            if new_state == RobotState.ALERT:
                # Start timer on hown long ALERT has been active
                self.alert_timer = time.time()
            self.prev_state = self.current_state
            self.current_state = new_state

    def handle_detection_timer(self, current_time, timer_name, gesture_duration, next_state):
        this_timer = getattr(self, timer_name)
        if this_timer is None:
            setattr(self, timer_name, current_time)
        elif current_time - this_timer > gesture_duration:
            self.transition_to(next_state)
            setattr(self, timer_name, None)

    def Resting(self):
        self.transition_to(RobotState.STANDBY)

    def evaluate(self, INDEX_FINGER, TWO_FINGERS, THREE_FINGERS, SHAKA_DETECTED):
        current_time = time.time()

        # If State is Approach
        # Continue Approaching or transition to ACTIVE
        if self.current_state == RobotState.APPROACH:
            if SHAKA_DETECTED: 
                # Continue approaching (zoom)
                return
            else:
                self.transition_to(RobotState.ACTIVE)

        # If State is Standby
        # Transition to ALERT if Index Finger deteceted
        elif self.current_state == RobotState.STANDBY:
            if INDEX_FINGER:
                self.handle_detection_timer(current_time, 'first_gesture_timer' , self.gesture_duration, RobotState.ALERT)
        
        # If State is ALERT
        # Transition to ACTIVE/HOLD if Two Fingers detected (depends on previous state)
        elif self.current_state == RobotState.ALERT:
            if TWO_FINGERS:
                if self.prev_state == RobotState.STANDBY: self.transition_to(RobotState.ACTIVE)
                elif self.prev_state == RobotState.ACTIVE: self.transition_to(RobotState.HOLD)
                elif self.prev_state == RobotState.HOLD: self.transition_to(RobotState.ACTIVE)

            elif current_time - self.alert_timer > self.reset_timeout:
                self.transition_to(self.prev_state)
       
        # If State is ACTIVE
        # Transition to STANDBY if Three Fingers detected
        # Transition to ALERT if Index Finger detected
        # Transition to APPROACH if Shaka detected
        elif self.current_state == RobotState.ACTIVE:
            if THREE_FINGERS:
                self.handle_detection_timer(current_time, 'reset_timer' , self.reset_duration, RobotState.STANDBY)                
            elif INDEX_FINGER:
                self.handle_detection_timer(current_time, 'first_gesture_timer' , self.gesture_duration, RobotState.ALERT)
            elif SHAKA_DETECTED:
                logging.debug('SHAKA_DETECTED - Attempt to change state')
                self.handle_detection_timer(current_time, 'zoom_gesture_timer', self.zoom_duration, RobotState.APPROACH)

        # If State is HOLD
        # Transition to ALERT if Index Finger is detected
        elif self.current_state == RobotState.HOLD:
            if INDEX_FINGER:
                self.handle_detection_timer(current_time, 'first_gesture_timer' , self.gesture_duration, RobotState.ALERT)
        
        # Reset Timers
        if not SHAKA_DETECTED:  self.zoom_gesture_timer = None
        if not INDEX_FINGER:    self.first_gesture_timer = None
        if not THREE_FINGERS:   self.reset_timer = None
            
    



         