#include "LEDControl.h"

CRGB leds[NUM_LEDS];  // Define LED array

void processLEDCommand(String data) {
    int firstCommaIndex = data.indexOf(',');
    int secondCommaIndex = data.indexOf(',', firstCommaIndex + 1);
    int thirdCommaIndex = data.indexOf(',', secondCommaIndex + 1);
    int fourthCommaIndex = data.indexOf(',', thirdCommaIndex + 1);
    int fifthCommaIndex = data.indexOf(',', fourthCommaIndex + 1);
    // Extract action command
    String action = data.substring(firstCommaIndex + 1, secondCommaIndex);

    if (action == "ON") {
        int r = data.substring(secondCommaIndex + 1, thirdCommaIndex).toInt();
        int g = data.substring(thirdCommaIndex + 1, fourthCommaIndex).toInt();
        int b = data.substring(fourthCommaIndex + 1, fifthCommaIndex).toInt();
        int brightness = data.substring(fifthCommaIndex + 1).toInt();  
        if (brightness == 0) brightness = 255;  // Default to max brightness if missing
        turnOnAllLEDs(CRGB(r, g, b), brightness);
    }
    else if (action == "OFF") {
        turnOffLEDs();
    }
    else if (action == "BREATH") {
        int r = data.substring(secondCommaIndex + 1, thirdCommaIndex).toInt();
        int g = data.substring(thirdCommaIndex + 1, fourthCommaIndex).toInt();
        int b = data.substring(fourthCommaIndex + 1, fifthCommaIndex).toInt();
        int brightness = data.substring(fifthCommaIndex + 1).toInt(); 
        if (brightness == 0) brightness = 255;  // Default to max brightness if missing
        // Set global breathing parameters
        breathColor = CRGB(r, g, b);
        breathBrightness = brightness;  // Store brightness for the loop
        breathingEffect(CRGB(r, g, b), brightness);
    }
    else if (action == "STATE" ){
        int r = data.substring(secondCommaIndex + 1, thirdCommaIndex).toInt();
        int g = data.substring(thirdCommaIndex + 1, fourthCommaIndex).toInt();
        int b = data.substring(fourthCommaIndex + 1, fifthCommaIndex).toInt();
        int brightness = data.substring(fifthCommaIndex + 1).toInt();  
        if (brightness == 0) brightness = 255;  // Default to max brightness if missing
        turnIntermediateState(CRGB(r, g, b), brightness);
    }
}

void setupLEDs() {
    FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS);
    FastLED.clear();
    FastLED.show();
}

// Turn on all LEDs with specified color and brightness
void turnOnAllLEDs(CRGB color, uint8_t brightness) {
    FastLED.setBrightness(brightness);
    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = color;
    }
    FastLED.show();
}

// Turn off all LEDs
void turnOffLEDs() {
    FastLED.clear();
    FastLED.show();
}

// Breathing effect (smooth brightness pulsing)
void breathingEffect(CRGB color, uint8_t maxBrightness) {
    static int brightness = 0;
    static bool increasing = true;

    FastLED.setBrightness(brightness);
    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = color;
    }
    FastLED.show();
    delay(40);
    // Adjust brightness for breathing effect
    if (increasing) {
        brightness += 2;
        if (brightness >= maxBrightness) {
            brightness = maxBrightness;
            increasing = false;
        }  
    } else {
        brightness -= 2;
        if (brightness <= 30) {
            brightness = 30;
            increasing = true;  // Minimum brightness
        }
    }
}

void setSingleLED(int index, CRGB color) {
    if (index >= 0 && index < NUM_LEDS) {  // Ensure index is valid
        leds[index] = color;  // Change only the selected LED
        FastLED.show();  // Update LEDs
    }
}

void turnIntermediateState(CRGB color, uint8_t brightness) {
   // Turn all LEDs one color and single LED color Blue
    FastLED.setBrightness(brightness);
    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = color;
    }
    FastLED.show();
    setSingleLED(17,CRGB::Blue);
}

