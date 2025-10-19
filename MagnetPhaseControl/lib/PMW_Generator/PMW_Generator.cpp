#include <Arduino.h>
#include "driver/mcpwm.h"
#include "PMW_Generator.h"

void setup_pwm(float duty, float pmw_freq) {
    // Configure MCPWM
    mcpwm_config_t pwm_config = {
        .frequency = pmw_freq,
        .cmpr_a = duty,
        .cmpr_b = duty,
        .duty_mode = MCPWM_DUTY_MODE_0,
        .counter_mode = MCPWM_UP_COUNTER,
    };

    // Initialize timers (use TIMER_0 and TIMER_1 so we can phase-shift between them)
    mcpwm_init(MCPWM_UNIT_0, MCPWM_TIMER_0, &pwm_config);
    mcpwm_init(MCPWM_UNIT_0, MCPWM_TIMER_1, &pwm_config);

    // Attach GPIOs
    mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM0A, PWM1_GPIO); // timer0 A
    mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM1A, PWM2_GPIO); // timer1 A (use TIMER_1 A pin), adjust if using different operator

    // Start timers
    mcpwm_start(MCPWM_UNIT_0, MCPWM_TIMER_0);
    mcpwm_start(MCPWM_UNIT_0, MCPWM_TIMER_1);

    printf("Dual PWM ready on GPIO %d and %d, Fault pin %d\n", PWM1_GPIO, PWM2_GPIO, FAULT_PIN);
}

bool handle_fault() {
    if (digitalRead(FAULT_PIN) == LOW) {
      mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_A, 0);
      mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_B, 0);
      Serial.println("ERROR: Fault detected! PWM outputs disabled.");
      return true;
    }
    return false;
}

void set_phase_degrees(float deg, uint32_t timer_period_ticks) {
    // convert degrees to timer ticks (0..timer_period_ticks-1)
    uint32_t phase_ticks = (uint32_t)((deg / 360.0f) * (float)timer_period_ticks) % timer_period_ticks;

    // Apply phase to TIMER_1 so it is shifted relative to TIMER_0.
    // Stop timers, set phase on TIMER_1, then restart so the new phase takes effect.
    mcpwm_stop(MCPWM_UNIT_0, MCPWM_TIMER_0);
    mcpwm_stop(MCPWM_UNIT_0, MCPWM_TIMER_1);

    mcpwm_timer_set_phase(MCPWM_UNIT_0, MCPWM_TIMER_1, phase_ticks);

    mcpwm_start(MCPWM_UNIT_0, MCPWM_TIMER_0);
    mcpwm_start(MCPWM_UNIT_0, MCPWM_TIMER_1);
}