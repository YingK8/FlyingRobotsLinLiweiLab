#include <Arduino.h>
#include "driver/mcpwm.h"

#define PWM1_GPIO 18
#define PWM2_GPIO 19
#define PWM_FREQ 300
#define FAULT_PIN 4

void setup_pwm(float duty, float pmw_freq);
bool handle_fault();
void set_phase_degrees(float deg, uint32_t timer_period_ticks);