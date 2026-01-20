#include "PhasePWM.h"
#include "esp_log.h"

static const char* TAG = "PhasePWM";

PhasePWM::PhasePWM(const int* channel_pins, size_t total_channels, int sync_pin, int pwm_freq_hz, bool is_main) 
    : _channel_pins(channel_pins), 
      _total_channels(total_channels), 
      _sync_pin(sync_pin), 
      _pwm_freq(pwm_freq_hz),
      _is_main(is_main)
{
    if (_total_channels > MAX_CHANNELS) {
        _total_channels = MAX_CHANNELS;
        ESP_LOGW(TAG, "Channel count limited to MAX_CHANNELS (%d)", MAX_CHANNELS);
    }

    // Initialize state arrays
    for(size_t i=0; i<MAX_CHANNELS; i++) {
        _stored_duties[i] = 0.0f;
        _stored_phases[i] = 0.0f;
    }
}

bool PhasePWM::begin() {
    ESP_LOGI(TAG, "Initializing PhasePWM. Role: %s, Channels: %d, Freq: %d Hz", 
             _is_main ? "MAIN" : "CLIENT", _total_channels, _pwm_freq);
    // 1. Configure Sync Pin
    // Regardless of role, the Sync Pin is an INPUT to the MCPWM module.
    // MAIN: Used to bridge Group 0 -> Group 1 (if > 3 channels)
    // CLIENT: Used to receive sync signal from Main Node
    if (_sync_pin >= 0) {
        gpio_config_t io_conf = {};
        io_conf.intr_type = GPIO_INTR_DISABLE;
        io_conf.mode = GPIO_MODE_INPUT;
        io_conf.pin_bit_mask = (1ULL << _sync_pin);
        io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
        io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
        gpio_config(&io_conf);
    }

    // 2. Calculate Period Ticks
    _period_ticks = _timer_resolution / _pwm_freq;

    // 3. Create Timers
    for (size_t i = 0; i < _total_channels; i++) {
        mcpwm_timer_config_t timer_cfg = {};
        timer_cfg.group_id = i / 3; 
        timer_cfg.clk_src = MCPWM_TIMER_CLK_SRC_DEFAULT;
        timer_cfg.resolution_hz = _timer_resolution;
        timer_cfg.period_ticks = _period_ticks;
        timer_cfg.count_mode = MCPWM_TIMER_COUNT_MODE_UP;
        
        if (mcpwm_new_timer(&timer_cfg, &_timers[i]) != ESP_OK) {
            ESP_LOGE(TAG, "Failed to create timer for channel %d", i);
            return false;
        }
    }

    // 4. Create Sync Sources

    // Source A: Internal Timer Event (From Timer 0) 
    // Used inside a MAIN node for Group 0 slaves
    mcpwm_timer_sync_src_config_t timer_sync_cfg = {};
    timer_sync_cfg.timer_event = MCPWM_TIMER_EVENT_EMPTY; 
    if (mcpwm_new_timer_sync_src(_timers[0], &timer_sync_cfg, &_sync_source_internal) != ESP_OK) return false;

    // Source B: External GPIO Event (From Sync Pin)
    // Used by CLIENT nodes (all groups) and MAIN node (Group 1)
    if (_sync_pin >= 0) {
        mcpwm_gpio_sync_src_config_t gpio_sync_cfg = {};
        gpio_sync_cfg.group_id = 0; // Create for Group 0 initially, but handle is global usually
        gpio_sync_cfg.gpio_num = (gpio_num_t)_sync_pin;
        gpio_sync_cfg.flags.active_neg = false; // Sync on rising edge
        
        // Note: You might need separate sync sources for Group 0 and Group 1 hardware 
        // if the IDF version enforces strict group affinity, but usually GPIO sync is flexible.
        // We create one handle.
        if (mcpwm_new_gpio_sync_src(&gpio_sync_cfg, &_sync_source_external) != ESP_OK) {
            ESP_LOGE(TAG, "Failed to create external sync source");
            return false;
        }
    }

    // 5. Apply Sync Logic based on Role
    
    if (_is_main) {
        // --- MAIN MODE ---
        // Timer 0 is the Master (Free running, no sync source set)
        
        // Timers 1 & 2 (Group 0) sync to Timer 0 internally
        for (size_t i = 1; i < 3 && i < _total_channels; i++) {
            mcpwm_timer_sync_phase_config_t sync_cfg = {};
            sync_cfg.sync_src = _sync_source_internal;
            sync_cfg.count_value = 0;
            sync_cfg.direction = MCPWM_TIMER_DIRECTION_UP;
            mcpwm_timer_set_phase_on_sync(_timers[i], &sync_cfg);
        }

        // Timers 3, 4, 5 (Group 1) cannot see Timer 0 internally.
        // They must sync via the external pin (Loopback Ch0 -> SyncPin)
        for (size_t i = 3; i < _total_channels; i++) {
            if (_sync_source_external) {
                mcpwm_timer_sync_phase_config_t sync_cfg = {};
                sync_cfg.sync_src = _sync_source_external;
                sync_cfg.count_value = 0;
                sync_cfg.direction = MCPWM_TIMER_DIRECTION_UP;
                mcpwm_timer_set_phase_on_sync(_timers[i], &sync_cfg);
            }
        }
    } 
    else {
        // --- CLIENT MODE ---
        // All timers sync to the external pin (connected to Main's Ch0)
        for (size_t i = 0; i < _total_channels; i++) {
            if (_sync_source_external) {
                mcpwm_timer_sync_phase_config_t sync_cfg = {};
                sync_cfg.sync_src = _sync_source_external;
                sync_cfg.count_value = 0; // Reset to 0 on sync pulse
                sync_cfg.direction = MCPWM_TIMER_DIRECTION_UP;
                mcpwm_timer_set_phase_on_sync(_timers[i], &sync_cfg);
            }
        }
    }

    // 6. Setup Operators, Comparators, and Generators
    for (size_t i = 0; i < _total_channels; i++) {
        // -- Operator --
        mcpwm_operator_config_t oper_cfg = {};
        oper_cfg.group_id = i / 3;
        if (mcpwm_new_operator(&oper_cfg, &_operators[i]) != ESP_OK) return false;
        if (mcpwm_operator_connect_timer(_operators[i], _timers[i]) != ESP_OK) return false;

        // -- Comparator --
        mcpwm_comparator_config_t cmpr_cfg = {};
        cmpr_cfg.flags.update_cmp_on_tez = true;
        if (mcpwm_new_comparator(_operators[i], &cmpr_cfg, &_comparators[i]) != ESP_OK) return false;

        // -- Generator --
        mcpwm_generator_config_t gen_cfg = {};
        gen_cfg.gen_gpio_num = _channel_pins[i];
        if (mcpwm_new_generator(_operators[i], &gen_cfg, &_generators[i]) != ESP_OK) return false;

        // -- Waveform Logic --
        mcpwm_generator_set_action_on_timer_event(_generators[i],
            MCPWM_GEN_TIMER_EVENT_ACTION(MCPWM_TIMER_DIRECTION_UP, MCPWM_TIMER_EVENT_EMPTY, MCPWM_GEN_ACTION_HIGH));
        mcpwm_generator_set_action_on_compare_event(_generators[i],
            MCPWM_GEN_COMPARE_EVENT_ACTION(MCPWM_TIMER_DIRECTION_UP, _comparators[i], MCPWM_GEN_ACTION_LOW));

        // -- Initial Duty & Phase --
        setDuty(i, 0.0);
        
        // Important: For Client nodes, setPhase logic might need to apply immediately, 
        // but setPhase() logic below also checks roles/indices. 
        // We call it here to ensure internal arrays are init.
        setPhase(i, 0.0);
    }

    // 7. Enable and Start
    for (size_t i = 0; i < _total_channels; i++) {
        mcpwm_timer_enable(_timers[i]);
    }
    
    // Start Slaves first (Indices > 0)
    for (size_t i = 1; i < _total_channels; i++) {
         mcpwm_timer_start_stop(_timers[i], MCPWM_TIMER_START_NO_STOP);
    }
    
    // Start Master Timer (Index 0) last.
    // In CLIENT mode, this timer waits for the sync pulse anyway.
    // In MAIN mode, this timer starts immediately and drives the others.
    mcpwm_timer_start_stop(_timers[0], MCPWM_TIMER_START_NO_STOP);

    return true;
}

void PhasePWM::setPhase(int user_channel_index, float degrees) {
    if (user_channel_index >= (int)_total_channels || user_channel_index < 0) return;

    _stored_phases[user_channel_index] = degrees;

    // In MAIN mode, Channel 0 is the reference, so shifting it is non-sensical relative to itself.
    if (_is_main && user_channel_index == 0) return; 

    // In CLIENT mode, Channel 0 CAN be shifted relative to the incoming sync signal!
    
    mcpwm_timer_sync_phase_config_t sync_phase_config = {};
    uint32_t shift_ticks = degreesToTicks(degrees);
    
    // Calculate Lag phase
    if (shift_ticks == 0) {
        sync_phase_config.count_value = 0;
    } else {
        sync_phase_config.count_value = _period_ticks - shift_ticks;
    }
    
    sync_phase_config.direction = MCPWM_TIMER_DIRECTION_UP;

    // Select Sync Source based on Role and Index
    if (_is_main) {
        // MAIN MODE
        if (user_channel_index < 3) {
            sync_phase_config.sync_src = _sync_source_internal;
        } else {
            sync_phase_config.sync_src = _sync_source_external;
        }
    } 
    else {
        // CLIENT MODE
        // All clients use external sync
        sync_phase_config.sync_src = _sync_source_external;
    }

    // Apply only if source exists
    if (sync_phase_config.sync_src) {
        mcpwm_timer_set_phase_on_sync(_timers[user_channel_index], &sync_phase_config);
    }
}

void PhasePWM::setDuty(int user_channel_index, float percent) {
    if (user_channel_index >= (int)_total_channels || user_channel_index < 0) return;

    _stored_duties[user_channel_index] = percent;

    uint32_t duty_ticks = percentToTicks(percent);
    mcpwm_comparator_set_compare_value(_comparators[user_channel_index], duty_ticks);
}

uint32_t PhasePWM::degreesToTicks(float degrees) {
    while (degrees >= 360.0) degrees -= 360.0;
    while (degrees < 0.0) degrees += 360.0;
    return (uint32_t)((_period_ticks * degrees) / 360.0f);
}

uint32_t PhasePWM::percentToTicks(float percent) {
    if (percent > 100.0) percent = 100.0;
    if (percent < 0.0) percent = 0.0;
    return (uint32_t)(_period_ticks * (percent / 100.0f));
}

void PhasePWM::setFrequency(int pwm_freq_hz) {
    if (pwm_freq_hz <= 0) return;
    
    _pwm_freq = pwm_freq_hz;
    _period_ticks = _timer_resolution / _pwm_freq;

    for(size_t i=0; i<_total_channels; i++) {
        mcpwm_timer_set_period(_timers[i], _period_ticks);
    }

    // Re-apply Duty and Phase with new ticks
    for(size_t i=0; i<_total_channels; i++) {
        setDuty(i, _stored_duties[i]);
        setPhase(i, _stored_phases[i]); 
    }
}