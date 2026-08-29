#pragma once

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_timer.h"
#include <Arduino.h>

const int LED_PIN = 2;
const int NUM_CHANNELS = 4;

#if SWIM_SETUP
// DRV8874 in PH/EN mode (PMODE tied logic low): PWM_PINS drive PH (direction),
// CARRIER_PINS drive EN (amplitude). EN=0 is Brake, so 0% carrier is a real off.
// The driver makes PH's complement internally -- no external inverter -- so this
// rig builds with GATE_ACTIVE_LOW=0. Inputs take 0-100 kHz (datasheet fPWM).
const int PWM_FREQ = 5000;        // carrier (Hz)

// Not GPIO14: that is A_PWM_PIN in this map. GPIO_NUM_NC disables the button.
const gpio_num_t RESET_BUTTON_PIN = GPIO_NUM_13;
const gpio_num_t A_PWM_PIN = GPIO_NUM_14;
const gpio_num_t B_PWM_PIN = GPIO_NUM_32;
const gpio_num_t C_PWM_PIN = GPIO_NUM_33;
const gpio_num_t D_PWM_PIN = GPIO_NUM_27;

// Adafruit ESP32 Feather V2 labels: A0, A1, MISO, A5. NOT the A2/A3/A4 pins --
// those are GPIO 34/39/36, which are input-only on the ESP32 and cannot drive
// EN at all (see isOutputCapable below). Boot-strap pins 5/12/15 are avoided
// too: a boot-time high on EN would energize a coil before setup() runs.
const gpio_num_t A_CARRIER_PIN = GPIO_NUM_26;
const gpio_num_t B_CARRIER_PIN = GPIO_NUM_25; 
const gpio_num_t C_CARRIER_PIN = GPIO_NUM_21; 
const gpio_num_t D_CARRIER_PIN = GPIO_NUM_4; 

const gpio_num_t A_ADC_PIN = GPIO_NUM_NC;
const gpio_num_t B_ADC_PIN = GPIO_NUM_NC; 
const gpio_num_t C_ADC_PIN = GPIO_NUM_NC; 
const gpio_num_t D_ADC_PIN = GPIO_NUM_NC; 

#else
const int PWM_FREQ = 20000;       // carrier (Hz); VNH5019 PWM pin, build with GATE_ACTIVE_LOW=1

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
#endif

const gpio_num_t PWM_PINS[NUM_CHANNELS] =     {A_PWM_PIN,     B_PWM_PIN,      C_PWM_PIN,      D_PWM_PIN};
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {A_CARRIER_PIN, B_CARRIER_PIN,  C_CARRIER_PIN,  D_CARRIER_PIN};
const gpio_num_t ADC_PINS[NUM_CHANNELS] =     {A_ADC_PIN,     B_ADC_PIN,      C_ADC_PIN,      D_ADC_PIN};

// GPIO34-39 are input-only on the classic ESP32 -- no output driver at all --
// and GPIO6-11 are the SPI flash. A gate pin mapped to either is silently dead:
// LEDC attaches without error, nothing comes out of the pad, and the DRV8874's
// internal 100k pulldown parks that EN low, so the channel sits in Brake for the
// whole run while the logs look perfectly healthy.
constexpr bool isOutputCapable(gpio_num_t p) {
  return p == GPIO_NUM_NC ||
         (p <= GPIO_NUM_33 && !(p >= GPIO_NUM_6 && p <= GPIO_NUM_11));
}
static_assert(isOutputCapable(A_PWM_PIN) && isOutputCapable(B_PWM_PIN) &&
                  isOutputCapable(C_PWM_PIN) && isOutputCapable(D_PWM_PIN) &&
                  isOutputCapable(A_CARRIER_PIN) && isOutputCapable(B_CARRIER_PIN) &&
                  isOutputCapable(C_CARRIER_PIN) && isOutputCapable(D_CARRIER_PIN),
              "A PWM/CARRIER pin cannot drive an output: GPIO34-39 are input-only "
              "and GPIO6-11 are the SPI flash");

// The reset-button poll digitalReads its pin every loop. If it shares a pin with
// a coil gate, it reads that gate's own idle-HIGH level, debounces it as a press
// and halts the firmware ~30 ms after boot -- pins frozen, freq stuck at 0.
static_assert(RESET_BUTTON_PIN == GPIO_NUM_NC ||
              (RESET_BUTTON_PIN != A_PWM_PIN && RESET_BUTTON_PIN != B_PWM_PIN &&
                  RESET_BUTTON_PIN != C_PWM_PIN && RESET_BUTTON_PIN != D_PWM_PIN &&
                  RESET_BUTTON_PIN != A_CARRIER_PIN && RESET_BUTTON_PIN != B_CARRIER_PIN &&
                  RESET_BUTTON_PIN != C_CARRIER_PIN && RESET_BUTTON_PIN != D_CARRIER_PIN),
              "RESET_BUTTON_PIN collides with a gate pin: the button poll will "
              "read the gate and block the run at boot");
