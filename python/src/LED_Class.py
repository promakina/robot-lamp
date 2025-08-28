import logging

from config_loader import load_parameters
rp = load_parameters("robot_parameters.py")

class LED:
    def __init__(self, serial_port):
        self.ser = serial_port
        self.state = None
        self.colors = rp.colors

    def getColor(self, state):
        # If 'state' is not in state_params, default to a dictionary with 'color': 'WHITE'
        default = {'color': 'WHITE'}
        return rp.state_params.get(state, default)['color']
    
    def getBrightness(self, state):
        default = {'brightness': 100}  # Default brightness value
        return rp.state_params.get(state, default)['brightness']
    
    def update_state(self, new_state):
        self.state = new_state

    def Single(self):
        this_state = 'single_led'
        if this_state == self.state: return

        self.update_state(this_state)
        self.send_LED_message("EVENT_DETECTED")

    def OFF(self):
        this_state = 'off'
        if this_state == self.state: return

        self.update_state(this_state)
        self.send_LED_message("OFF")

    def STAY(self):
        this_state = 'stay'
        if this_state == self.state: return

        self.update_state(this_state)
        self.send_LED_message("ON", self.getColor(this_state), self.getBrightness(this_state))

    def Standby(self):
        this_state = 'standby'
        if this_state == self.state: return

        self.update_state(this_state)
        self.send_LED_message("BREATH", self.getColor(this_state), self.getBrightness(this_state))

    def Tracking_Enabled(self):
        this_state = 'tracking'
        if this_state == self.state: return

        self.update_state(this_state)
        self.send_LED_message("ON", self.getColor(this_state), self.getBrightness(this_state))

    def Tracking_Blink(self):
        this_state = 'blink'
        if this_state == self.state: return

        self.update_state(this_state)
        self.send_LED_message("ON", "GREEN_DARK", 20)

    def State1_NotTracking(self):
        this_state = 'state1_notTracking'
        if this_state == self.state: return

        self.update_state(this_state)
        self.send_LED_message("STATE",self.getColor(this_state),self.getBrightness(this_state))

    def State1_Tracking(self):
        this_state = 'state1_Tracking'
        if this_state == self.state: return

        self.update_state(this_state)
        self.send_LED_message("STATE",self.getColor(this_state), self.getBrightness(this_state))

    def send_LED_message(self, action, color = "WHITE", brightness = 100):
        # Convert color names to RGB values
        color = self.colors.get(color.upper(), "255,255,255")  # Default to WHITE if invalid color

        # Map brightness from 0-100 to 0-255
        brightness = max(1, int((brightness / 100) * 255))  # Prevent 0 (off state)

        msg = f'LED,{action},{color},{brightness}\n'
        self.ser.write(msg.encode('utf-8'))