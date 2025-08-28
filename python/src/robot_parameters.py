# ESP32 COM PORT
COM_PORT = 'COM5'

# CAMERA
CAMERA_INDEX = 0

# Define link lengths
j1_height = 123.3   # Z height of J1 (distance from J1 to ground)
link_a = 118    # J1 to J2
link_b = 30     # J2 to J3 

# Camera
ADJUST_CAMERA_RESOLUTION = False
camera_resolution_width = 1280
camera_resolution_height = 720

# Detection
RESET_TIMEOUT = 2.0                 # Timeout to reset detection
FIRST_GESTURE_DURATION = 0.6        # Seconds to hold gesture to activate tracking
ZOOM_GESTURE_DURATION = 1         # Seconds to hold gesture to activate zoom
RESET_GESTURE_DURATION = 0.8        # Seconds to hold gesture to activate reset
DEADBAND = 50                       # Threshold to enable tracking, in pixels

# Display
FONT_SIZE = 0.6
FONT_THICKNESS = 2

######################################
# LED
######################################
# set colors RGB values
colors = {
    "NEUTRAL": "255,220,190",
    "SOFT": "255,180,100",
    "WARM": "239,192,112",
    "WHITE": "255,255,255",
    "RED": "255,0,0",
    "GREEN": "0,255,0",
    "GREEN_DARK": "50,150,15",
    "BLUE": "0,0,255",
    "YELLOW": "255,255,0",
    "CYAN": "0,255,255",
    "MAGENTA": "255,0,255"
}

# set state color and brightness
state_params = {
    'tracking':             {'color': 'SOFT', 'brightness':     20},
    'state1_notTracking':   {'color': 'WARM', 'brightness':     100},
    'state1_Tracking':      {'color': 'SOFT', 'brightness':     20},
    'standby':              {'color': 'BLUE', 'brightness':     40},
    'stay':                 {'color': 'WARM', 'brightness':     100}
}