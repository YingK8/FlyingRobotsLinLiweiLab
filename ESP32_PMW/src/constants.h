#pragma once

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_timer.h"
#include <Arduino.h>

const int PWM_FREQ = 20000;       // carrier (Hz)

// Host link speed. 115200 put 29 bytes -- one `mag=` or `az=` line -- 2.5 ms on the
// wire, which is longer than a 500 Hz control period and swamped the whole reason for
// raising the control clock: at 200 Hz the mean command age was 2.5 ms and the wire
// added another 2.5, so the clock could only ever fix half the delay. At 921600 those
// same 29 bytes take 0.31 ms. Bandwidth was never the constraint -- the host's deadbands
// hold the line at 92 B/s, 0.8% of 115200 (`controller/control/theory.md` 19.10) --
// latency was.
//
// MUST match `controller/control/link.SerialComm.BAUD` and `platformio.ini`'s
// `monitor_speed`. There is no handshake and no ack, so a mismatch is silent: the
// firmware simply never parses a command and the coils sit at their last value.
const int SERIAL_BAUD = 921600;

const int LED_PIN = 2;
const int NUM_CHANNELS = 4;

// Momentary reset button: wired to drive this pin to 3V3 when pressed (active
// HIGH). GPIO14 has an internal pulldown, so it idles LOW with no external
// resistor. See reset_button.h.
const gpio_num_t RESET_BUTTON_PIN = GPIO_NUM_14;

#if SWIM_SETUP
const gpio_num_t A_PWM_PIN = GPIO_NUM_27;
const gpio_num_t B_PWM_PIN = GPIO_NUM_12;
const gpio_num_t C_PWM_PIN = GPIO_NUM_15;
const gpio_num_t D_PWM_PIN = GPIO_NUM_33;

const gpio_num_t A_CARRIER_PIN = GPIO_NUM_NC;
const gpio_num_t B_CARRIER_PIN = GPIO_NUM_NC; 
const gpio_num_t C_CARRIER_PIN = GPIO_NUM_NC; 
const gpio_num_t D_CARRIER_PIN = GPIO_NUM_NC; 

const gpio_num_t A_ADC_PIN = GPIO_NUM_NC;
const gpio_num_t B_ADC_PIN = GPIO_NUM_NC; 
const gpio_num_t C_ADC_PIN = GPIO_NUM_NC; 
const gpio_num_t D_ADC_PIN = GPIO_NUM_NC; 

#else
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
#endif

const gpio_num_t PWM_PINS[NUM_CHANNELS] =     {A_PWM_PIN,     B_PWM_PIN,      C_PWM_PIN,      D_PWM_PIN};
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {A_CARRIER_PIN, B_CARRIER_PIN,  C_CARRIER_PIN,  D_CARRIER_PIN};
const gpio_num_t ADC_PINS[NUM_CHANNELS] =     {A_ADC_PIN,     B_ADC_PIN,      C_ADC_PIN,      D_ADC_PIN};