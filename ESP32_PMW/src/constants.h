#pragma once

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_timer.h"
#include <Arduino.h>

const int PWM_FREQ = 20000;       // carrier (Hz)

const int LED_PIN = 2;
const int NUM_CHANNELS = 4;

// Momentary reset button: wired to drive this pin to 3V3 when pressed (active
// HIGH). GPIO14 has an internal pulldown, so it idles LOW with no external
// resistor. See reset_button.h.
const gpio_num_t RESET_BUTTON_PIN = GPIO_NUM_14;

const gpio_num_t A_PWM_PIN = GPIO_NUM_32;
const gpio_num_t B_PWM_PIN = GPIO_NUM_25;
const gpio_num_t C_PWM_PIN = GPIO_NUM_18;
const gpio_num_t D_PWM_PIN = GPIO_NUM_22;

const gpio_num_t A_CARRIER_PIN = GPIO_NUM_33;
const gpio_num_t B_CARRIER_PIN = GPIO_NUM_26; 
const gpio_num_t C_CARRIER_PIN = GPIO_NUM_19; 
const gpio_num_t D_CARRIER_PIN = GPIO_NUM_23; 

const gpio_num_t A_ADC_PIN = GPIO_NUM_36;
const gpio_num_t B_ADC_PIN = GPIO_NUM_39; 
const gpio_num_t C_ADC_PIN = GPIO_NUM_34; 
const gpio_num_t D_ADC_PIN = GPIO_NUM_35; 

const gpio_num_t PWM_PINS[NUM_CHANNELS] =     {A_PWM_PIN,     B_PWM_PIN,      C_PWM_PIN,      D_PWM_PIN};
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {A_CARRIER_PIN, B_CARRIER_PIN,  C_CARRIER_PIN,  D_CARRIER_PIN};
const gpio_num_t ADC_PINS[NUM_CHANNELS] =     {A_ADC_PIN,     B_ADC_PIN,      C_ADC_PIN,      D_ADC_PIN};