# Robot Class

from config_loader import load_parameters

rp = load_parameters("robot_parameters.py")
mp = load_parameters("motor_parameters.py")

import logging
import math
import fwd_kinematics as fk
import inverse_kinematics as ik
import numpy as np
import time


class Robot:
    def __init__(self, serial_port):
        self.ser = serial_port
        self.drive_timer = time.time()
        self.drive_interval = 0.05
        self.angles = None
        self.move_to_angles = None
        self.home_angles = mp.home_angles
        self.rest_angles = mp.rest_angles
        self.integral_error_x = 0
        self.integral_error_y = 0
        self.offset_x_prev = 0
        self.offset_y_prev = 0
        self.alpha = mp.alpha # Filter factor for offsets

    def check_for_nan(self, angles):
        """
        Check if any value in the list of angles is NaN.

        Parameters:
        angles (list): A list of angles to check.

        Returns:
        bool: True if any value in the list is NaN, False otherwise.
        """
        for angle in angles:
            if math.isnan(angle):
                return True
        return False

    def clamp_angles_to_limits(self, angles):
        """
        Clamps each joint angle within its min/max limit.
        Returns the adjusted list of angles.
        """
        clamped_angles = []
        for index, angle in enumerate(angles):
            min_limit = mp.ANGLE_MIN_LIMITS[index]
            max_limit = mp.ANGLE_MAX_LIMITS[index]

            if angle < min_limit:
                logging.debug(f"Angle below min limit - Joint: {index}, original: {angle}, clamped: {min_limit}")
                clamped_angles.append(min_limit)
            elif angle > max_limit:
                logging.debug(f"Angle above max limit - Joint: {index}, original: {angle}, clamped: {max_limit}")
                clamped_angles.append(max_limit)
            else:
                clamped_angles.append(angle)

        return clamped_angles

    def driveHome(self, t = 1000):
        logging.info('DRIVING_TO_HOME')
        self.move_to_angles = self.home_angles
        self.driveRobot(t = t, bypassTimer=True)

    def driveToRest(self, t = 1000):
        logging.info('DRIVING_TO_REST')
        self.move_to_angles = self.rest_angles
        self.driveRobot(t = t, bypassTimer=True)

    def findServoSpeed(self):
        '''
        Function will evaluate current angles (self.angles)
        against target angles (self.move_to_angles)
        and find a suitable speed to send to the ESP32

        Speed is given as time to reach target angle, in ms
        All Joint to be driven at the same speed
        '''
        # Define angle difference limits (example: smallest and largest angles you expect)
        min_angle_diff = 0    # no movement
        max_possible_angle_diff = 180  # adjust according to your robot's joint limits
        # Speed range limits
        min_speed = mp.min_speed
        max_speed = mp.max_speed
        
        # Find angle each joint needs to move
        abs_diff = [abs(a - b) for a, b in zip(self.angles, self.move_to_angles)]
        
        max_angle_diff = max(abs_diff)

        # Mapping function (linear interpolation)
        mapped_speed = ((max_angle_diff - min_angle_diff) / (max_possible_angle_diff - min_angle_diff)) * (max_speed - min_speed) + min_speed
     
        return round(mapped_speed)

    def driveRobot(self, t = 0, bypassTimer = False):
        # First Check if robot needs to be moved
        if self.angles == self.move_to_angles: 
            #logging.debug('Robot_Does_Not_Need_To_Move')
            return
        
        # Avoid sending messages to fast
        if not bypassTimer and time.time() - self.drive_timer < self.drive_interval:
            logging.debug('Timer_Not_Reached')
            return
        
        if (t==0): t = self.findServoSpeed()
        #logging.debug(f'DriveRobot - Speed Set to {t}')

        self.drive_timer = time.time()      # Update timer
        self.angles = self.move_to_angles   # Update angles
        self.send_message(t)                # Send to ESP32

    def send_message(self, t=0):
        angles_as_floats = [round(float(angle), 0) for angle in self.angles]
        msg =   str(angles_as_floats[0]) + "," + str(angles_as_floats[1]) + "," + \
                str(angles_as_floats[2]) + "," + str(angles_as_floats[3]) + "," + str(t)
        out_message = msg + "\n"  # Add newline to indicate end of message
        self.ser.write(out_message.encode('utf-8'))
        logging.info(f'Message to ESP32: {msg}')

    def find_angles(self, new_offset_x, new_offset_y):
        '''
        Used for tracking mode
        Proportional & Integral control
        '''
        logging.info('Evaluating_Offsets | -find_angles- function')
        ###############################################
        # ---------- Check if within deadband --------
        if abs(new_offset_x) > rp.DEADBAND or abs(new_offset_y) > rp.DEADBAND: 
            logging.debug('Result: Outside DEADBAND')
        else: 
            logging.info('Result: Inside DEADBAND')
            return 
        ###############################################
        ###############################################

        # Apply filter to offsets
        offset_x = self.alpha * new_offset_x  + (1- self.alpha) * self.offset_x_prev
        offset_y = self.alpha * new_offset_y  + (1- self.alpha) * self.offset_y_prev
        # Update offsets to store
        self.offset_x_prev = offset_x
        self.offset_y_prev = offset_y

        #logging.debug(f'OFFSET_X - new: {new_offset_x}, filtered: {offset_x}')
        #logging.debug(f'OFFSET_Y - new: {new_offset_y}, filtered: {offset_y}')

        new_angles = self.move_to_angles  # make sure to make separate list
        
        # Find error
        errorX = abs(offset_x) - rp.DEADBAND
        errorY = abs(offset_y) - rp.DEADBAND
        # Accumulate integral error (avoid excessive accumulation)
        self.integral_error_x += errorX
        self.integral_error_x = max(min(self.integral_error_x, mp.integral_max), -mp.integral_max)
        self.integral_error_y += errorY
        self.integral_error_y = max(min(self.integral_error_y, mp.integral_max), -mp.integral_max)
        # Find step value
        Xstep_value = mp.kp_x * errorX + mp.ki_x * self.integral_error_x
        Ystep_value = mp.kp_y * errorY + mp.ki_y * self.integral_error_y
        # Apply step value in X
        if (offset_x < -rp.DEADBAND ): new_angles[0] += Xstep_value 
        elif (offset_x > rp.DEADBAND) : new_angles[0] -= Xstep_value
        # Apply step value in Y
        if (offset_y < -rp.DEADBAND ): 
            ystep = Ystep_value
        elif (offset_y > rp.DEADBAND) : 
            ystep = -Ystep_value
        else: ystep = 0

        new_angles[3] += ystep # Apply step to J3
        
        # Final check
        if not self.check_for_nan(new_angles): # Check if any nan first
            self.move_to_angles = self.clamp_angles_to_limits(new_angles)

    def apply_zoom(self, orientation):
        '''
        Apply 'dx' step to zoom in or zoom out
        'dx' is step in X direction, positive or negative
        Returns new joint angles after applying step
        '''
        filter_const = mp.alpha_zoom
        tolerance = 0.1 # Tolerance for calculation of next point
        logging.info("")
        logging.info(f'Enter_Zoom_Function, {orientation}')
        logging.debug(f'ZOOM | Current_Angles: {self.angles}')
        
        # Find absolute angle of J3
        # This is angle with respect of vertical
        # CCW is positive, CW is negative
        J3_global = self.move_to_angles[1] + self.move_to_angles[2] + self.move_to_angles[3]

        # Amount Position will change in X direction
        step_size = mp.step
        if orientation == 'IN':dx = step_size
        else: dx = -step_size

        # Start calculating
        base_height = rp.j1_height # z distance of J1 from ground
        # Find current position
        x,y,z = fk.fwd_kin(*self.move_to_angles)
        logging.debug(f'Zoom | Current_Position: {x}, {y}, {z}')
        
        # Rotate x coordinate such that y = 0
        x_temp_y0 = np.sqrt(x**2 + y**2)
        if x < 0: x_temp_y0 = -x_temp_y0 # take into account the sign

        # Find theta angle, this is angle of camera orientation
        # Consider J3=0 to be 90 deg, camera is facing perpendicular to horizonal
        theta = self.move_to_angles[3]
        
        # Find new XZ
        x_temp = x_temp_y0 + dx
        dz = np.tan(np.deg2rad(theta)) * dx
        z_temp = z + dz - base_height

        # Evaluate if XZ is reachable
        max_length = rp.link_a + rp.link_b
        new_length = np.sqrt(x_temp**2 + z_temp**2)
        if new_length > max_length:
            logging.debug('Point is unreachable, finding new one')
            # Point is outside envelope
            # Need to find new set of point in the same direction
            beta_rad = np.arctan(z_temp/x_temp)   # Find angle with horizontal
            x_temp = (max_length - tolerance) * np.cos(beta_rad)
            z_temp = (max_length - tolerance) * np.sin(beta_rad)
            new_length = np.sqrt(x_temp**2 + z_temp**2)

        # Now find x_new point by rotating by J0
        gamma = self.move_to_angles[0]
        hipotenuse = x_temp
        x_new = round(hipotenuse * np.cos(np.deg2rad(gamma)),2)
        y_new = round(hipotenuse * np.sin(np.deg2rad(gamma)),2)
        z_new = round(z_temp + base_height,2)

        # Perform inverse kinematics to find Angles of new point
        #logging.debug(f'Zoom | Target_Position: x: {x_new}, y: {y_new}, z: {z_new}')
        new_angles = ik.inverse_kin(x_new, y_new, z_new) 
        new_angles = list(new_angles)           # convert tuple to list

        # For the last angle, J3, need to match global angle
        new_J3 = J3_global - new_angles[1] - new_angles[2]
        new_angles.append(new_J3)

        #new_angles.append(self.move_to_angles[3])    # Add last angle, IK only does J0, J1, J2
        #logging.debug(f'ZOOM_target_angles: {new_angles}')
        # Final checks
        if not self.check_for_nan(new_angles): # Check if any nan first
            new_angles = self.clamp_angles_to_limits(new_angles)
            for index, angle in enumerate(new_angles):
                new_angles[index] = filter_const * self.move_to_angles[index] + (1-filter_const) * angle
            logging.info(f'Zoom_Successful, updating move_to_angles: {new_angles}')
            self.move_to_angles = new_angles
            return

        logging.info(f'Zoom_NOT_Successful, no update')
        

