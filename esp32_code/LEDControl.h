#ifndef LEDCONTROL_H
#define LEDCONTROL_H

#include <FastLED.h>

#define LED_PIN 4          // GPIO pin connected to WS2812 DIN
#define NUM_LEDS 35        // Number of LEDs in the ring
#define LED_TYPE WS2812B
#define COLOR_ORDER GRB

extern CRGB leds[NUM_LEDS];  // Declare LED array
extern CRGB breathColor;
extern uint8_t breathBrightness;

void setupLEDs();
void processLEDCommand(String data);
void turnOnAllLEDs(CRGB color, uint8_t brightness);
void turnOffLEDs();
void breathingEffect(CRGB color, uint8_t brightness);
void setSingleLED(int index, CRGB color);
void turnIntermediateState(CRGB color, uint8_t brightness);

#endif
