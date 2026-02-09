#include <Arduino.h>
#include "soc/gpio_struct.h"

/**
 * ESP32 Direct Register Write
 * Works for Pins 0-31. 
 * (For pins 32-39, use GPIO.out1_w1ts.val and GPIO.out1_w1tc.val)
 */
void directWrite(uint8_t pin, uint8_t value) {
  if (pin < 32) {
    if (value) {
      GPIO.out_w1ts = (1ULL << pin); // Set HIGH
    } else {
      GPIO.out_w1tc = (1ULL << pin); // Set LOW
    }
  } else {
    // For pins 32 and above (ESP32 has up to 39 GPIOs)
    if (value) {
      GPIO.out1_w1ts.val = (1ULL << (pin - 32));
    } else {
      GPIO.out1_w1tc.val = (1ULL << (pin - 32));
    }
  }
}