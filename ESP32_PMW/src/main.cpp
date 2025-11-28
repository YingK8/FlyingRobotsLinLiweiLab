#include <Arduino.h>

/*
 * ESP32 MCPWM Phase Shifted Signal Generator
 * Based on ESP-IDF v5.x driver (Arduino ESP32 Core 3.0+)
 *
 * FEATURES:
 * - 3 Independent PWM Channels
 * - Adjustable Global Frequency
 * - Independent Duty Cycles
 * - Independent Phase Shifts (0-360 degrees)
 */

#include <Arduino.h>
#include "driver/mcpwm_prelude.h"

// ================= CONFIGURATION =================
#define SERIES_RESISTOR_OHMS 0 // Not needed for logic signals
// #define LED_BUILTIN 2

// Define Output Pins
const int PWM_PINS[] = {14, 15, 32}; 
const int NUM_CHANNELS = 3;

// MCPWM Configuration
#define MCPWM_TIMER_RESOLUTION_HZ 10000000 // 10MHz, 1 tick = 100ns
#define INITIAL_FREQ_HZ           300     // 1kHz initial frequency

// Global Handles
mcpwm_timer_handle_t    timers[NUM_CHANNELS];
mcpwm_oper_handle_t     operators[NUM_CHANNELS];
mcpwm_cmpr_handle_t     comparators[NUM_CHANNELS];
mcpwm_gen_handle_t      generators[NUM_CHANNELS];
mcpwm_sync_handle_t     sync_source = NULL; // The master sync source

// Current State Tracking
uint32_t current_freq_hz = INITIAL_FREQ_HZ;
float current_phases[NUM_CHANNELS] = {0.0, 120.0, 240.0}; // Initial phases
float current_duties[NUM_CHANNELS] = {50.0, 50.0, 50.0};  // Initial duties

// ================= HELPER FUNCTIONS =================

/**
 * Update the Global Frequency.
 * NOTE: When frequency changes, we must recalculate the tick values 
 * for phase shifts and duty cycles to maintain the same percentages/degrees.
 */
void setGlobalFrequency(uint32_t freq_hz) {
    if (freq_hz == 0) return;
    current_freq_hz = freq_hz;

    // Calculate new period in ticks
    uint32_t period_ticks = MCPWM_TIMER_RESOLUTION_HZ / current_freq_hz;

    // Update all timers
    for (int i = 0; i < NUM_CHANNELS; i++) {
        // Update Period
        mcpwm_timer_set_period(timers[i], period_ticks);
        
        // We must re-apply the phase shift logic because 'ticks' have changed
        // For channel 0 (Master), phase is always 0, no sync action needed usually.
        // For Slaves, we calculate the count value to load on sync.
        if (i > 0) {
            // Phase shift logic:
            // If we want a 90 deg lag, when Master is at 0, Slave should be at (Period * 90/360).
            // Actually, easier logic: When Master resets (Sync), load Slave with (Period * Phase / 360).
            // This effectively pushes the Slave 'ahead' or 'behind' in the count cycle.
            
            uint32_t phase_val = (uint32_t)((float)period_ticks * current_phases[i] / 360.0f);
            
            mcpwm_timer_sync_phase_config_t sync_phase_config = {};
            sync_phase_config.count_value = phase_val; 
            sync_phase_config.direction = MCPWM_TIMER_DIRECTION_UP;
            sync_phase_config.sync_src = sync_source; // Listen to Master
            
            mcpwm_timer_set_phase_on_sync(timers[i], &sync_phase_config);
        }

        // Re-apply Duty Cycle (Percent to new Ticks)
        uint32_t duty_ticks = (uint32_t)(period_ticks * (current_duties[i] / 100.0f));
        mcpwm_comparator_set_compare_value(comparators[i], duty_ticks);
    }
}

/**
 * Set Phase for a specific channel.
 * Channel 0 is the reference (Master), so its phase is technically always 0 relative to itself.
 * Changing Ch0 phase here won't do anything visible unless we added an external sync.
 */
void setPhase(int channel, float degrees) {
    if (channel >= NUM_CHANNELS || channel < 1) return; // Cannot shift Master (0)
    
    current_phases[channel] = degrees;
    
    uint32_t period_ticks = MCPWM_TIMER_RESOLUTION_HZ / current_freq_hz;
    uint32_t phase_val = (uint32_t)((float)period_ticks * degrees / 360.0f);

    mcpwm_timer_sync_phase_config_t sync_phase_config = {};
    sync_phase_config.count_value = phase_val;
    sync_phase_config.direction = MCPWM_TIMER_DIRECTION_UP;
    sync_phase_config.sync_src = sync_source;

    mcpwm_timer_set_phase_on_sync(timers[channel], &sync_phase_config);
}

/**
 * Set Duty Cycle for a specific channel (0-100%)
 */
void setDuty(int channel, float percent) {
    if (channel >= NUM_CHANNELS) return;
    
    current_duties[channel] = percent;
    uint32_t period_ticks = MCPWM_TIMER_RESOLUTION_HZ / current_freq_hz;
    uint32_t duty_ticks = (uint32_t)(period_ticks * (percent / 100.0f));
    
    mcpwm_comparator_set_compare_value(comparators[channel], duty_ticks);
}

// ================= SETUP =================

void setup() {
    Serial.begin(115200);
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

// ================= MAIN LOOP =================

void loop() {
    // DEMO: Sweep frequencies and phases
    
    // Serial.println("Phase Sweep Demo...");
    
    // // Sweep Phase of Channel 1 from 0 to 180
    // for(int p = 0; p <= 180; p+=5) {
    //     setPhase(1, (float)p);
    //     delay(50);
    // }
    
    Serial.println("Duty Cycle Sweep Demo...");
    // Sweep Duty of Channel 0
    for(int d = 10; d <= 90; d++) {
        setDuty(0, (float)d);
        delay(20);
    }

    for(int d = 90; d >= 10; d--) {
        setDuty(0, (float)d);
        delay(20);
    }

    for(int d = 10; d <= 90; d++) {
        setDuty(1, (float)d);
        delay(20);
    }

    for(int d = 90; d >= 10; d--) {
        setDuty(1, (float)d);
        delay(20);
    }
    
    
    // Serial.println("Frequency Jump...");
    // setGlobalFrequency(20000); // 20 kHz
    // delay(2000);
    // setGlobalFrequency(1000);  // Back to 1 kHz
    // delay(2000);
}