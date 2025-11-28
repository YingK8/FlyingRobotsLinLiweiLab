#include <Arduino.h>
#include "driver/mcpwm_prelude.h"

void setup_pmw() {
    Serial.println("Initializing MCPWM Phase Shift Demo...");

    // 1. Create Timers
    // We use one timer per channel to allow for phase shifting via synchronization
    mcpwm_timer_config_t timer_config = {};
    timer_config.group_id = 0;
    timer_config.clk_src = MCPWM_TIMER_CLK_SRC_DEFAULT;
    timer_config.resolution_hz = MCPWM_TIMER_RESOLUTION_HZ;
    timer_config.period_ticks = MCPWM_TIMER_RESOLUTION_HZ / INITIAL_FREQ_HZ;
    timer_config.count_mode = MCPWM_TIMER_COUNT_MODE_UP;

    for (int i = 0; i < NUM_CHANNELS; i++) {
        ESP_ERROR_CHECK(mcpwm_new_timer(&timer_config, &timers[i]));
    }

    // 2. Setup Synchronization (The Magic Part)
    // We make Timer 0 the "Master". It generates a signal when it reaches Zero (TEZ).
    mcpwm_timer_sync_src_config_t sync_src_config = {};
    sync_src_config.timer_event = MCPWM_TIMER_EVENT_EMPTY; // Sync when timer counts to zero
    ESP_ERROR_CHECK(mcpwm_new_timer_sync_src(timers[0], &sync_src_config, &sync_source));

    // Configure Slaves (Timer 1, 2...) to sync to Master
    // They will load a specific counter value when Master resets.
    for (int i = 1; i < NUM_CHANNELS; i++) {
        mcpwm_timer_sync_phase_config_t sync_phase_config = {};
        // Calculate initial phase offset in ticks
        uint32_t period = MCPWM_TIMER_RESOLUTION_HZ / INITIAL_FREQ_HZ;
        sync_phase_config.count_value = (uint32_t)(period * (current_phases[i] / 360.0));
        sync_phase_config.direction = MCPWM_TIMER_DIRECTION_UP;
        sync_phase_config.sync_src = sync_source;
        
        ESP_ERROR_CHECK(mcpwm_timer_set_phase_on_sync(timers[i], &sync_phase_config));
    }

    // 3. Create Operators, Comparators, and Generators
    for (int i = 0; i < NUM_CHANNELS; i++) {
        // Operator
        mcpwm_operator_config_t oper_config = {};
        oper_config.group_id = 0; // Same group as timer
        ESP_ERROR_CHECK(mcpwm_new_operator(&oper_config, &operators[i]));
        ESP_ERROR_CHECK(mcpwm_operator_connect_timer(operators[i], timers[i]));

        // Comparator
        mcpwm_comparator_config_t cmpr_config = {};
        cmpr_config.flags.update_cmp_on_tez = true; // Update duty on zero
        ESP_ERROR_CHECK(mcpwm_new_comparator(operators[i], &cmpr_config, &comparators[i]));
        
        // Initial Duty Set
        uint32_t period = MCPWM_TIMER_RESOLUTION_HZ / INITIAL_FREQ_HZ;
        uint32_t duty_ticks = (uint32_t)(period * (current_duties[i] / 100.0));
        ESP_ERROR_CHECK(mcpwm_comparator_set_compare_value(comparators[i], duty_ticks));

        // Generator (Pin)
        mcpwm_generator_config_t gen_config = {};
        gen_config.gen_gpio_num = PWM_PINS[i];
        ESP_ERROR_CHECK(mcpwm_new_generator(operators[i], &gen_config, &generators[i]));

        // 4. Set Generator Actions (PWM Logic)
        // Go High when Counter == 0 (Empty)
        // Go Low when Counter == Compare Value
        ESP_ERROR_CHECK(mcpwm_generator_set_action_on_timer_event(generators[i],
                MCPWM_GEN_TIMER_EVENT_ACTION(MCPWM_TIMER_DIRECTION_UP, MCPWM_TIMER_EVENT_EMPTY, MCPWM_GEN_ACTION_HIGH)));
        ESP_ERROR_CHECK(mcpwm_generator_set_action_on_compare_event(generators[i],
                MCPWM_GEN_COMPARE_EVENT_ACTION(MCPWM_TIMER_DIRECTION_UP, comparators[i], MCPWM_GEN_ACTION_LOW)));
    }

    // 5. Enable and Start Timers
    for (int i = 0; i < NUM_CHANNELS; i++) {
        ESP_ERROR_CHECK(mcpwm_timer_enable(timers[i]));
        ESP_ERROR_CHECK(mcpwm_timer_start_stop(timers[i], MCPWM_TIMER_START_NO_STOP));
    }

    Serial.println("MCPWM Running. Check pins with Logic Analyzer.");
}
