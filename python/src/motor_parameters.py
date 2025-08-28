'''
Set Motor/Joints Parameters
'''
# --------------------------------
# JOINT LIMITS
# Joint 1 contains 2 motors (MOTOR1A & MOTOR1B)
# --------------------------------

# HOME POSITION
# J0, J1, J2, J3 in Degrees
home_angles = [0,   0,    0,    0]
rest_angles = [0,   90, -90,    -45]


# JOINT 0
J0_MIN_LIMIT = -45
J0_MAX_LIMIT = 45

# JOINT 1
J1A_MIN_LIMIT = -45
J1A_MAX_LIMIT = 45
J1B_MIN_LIMIT = J1A_MIN_LIMIT 
J1B_MAX_LIMIT = J1A_MAX_LIMIT 

# JOINT 2
J2_MIN_LIMIT = -45
J2_MAX_LIMIT = 45

# JOINT 3
J3_MIN_LIMIT = -90
J3_MAX_LIMIT = 90

# Create lists for the min and max limits of all motors
ANGLE_MIN_LIMITS = [J0_MIN_LIMIT, 
                    J1A_MIN_LIMIT, 
                    J2_MIN_LIMIT,
                    J3_MIN_LIMIT ]

ANGLE_MAX_LIMITS = [J0_MAX_LIMIT, 
                    J1A_MAX_LIMIT, 
                    J2_MAX_LIMIT,
                    J3_MAX_LIMIT ]

# --------------------------------
# JOINT CONTROL VALUES
# --------------------------------

# Proportional Constants
kp_x = 0.015
kp_y = 0.015
# Integral Constants
ki_x = 0.01
ki_y = 0.01
# Prevent integral windup
integral_max = 50 
# Motor speeds, given in ms to reach target angle
# The program will interpolate between these values
# to find the speed for a specific angle
min_speed = 100
max_speed = 300 
# Low pass filter constant to apply to offsets
# This smoothes out input to PID controller
alpha = 0.3

# --------------------------------
# ZOOM CONTROL VALUES
# --------------------------------
# Apply filter for slower movement during zoom
alpha_zoom = 0.0    # Low Pass Filter for ZOOM
step = 3            # step size
