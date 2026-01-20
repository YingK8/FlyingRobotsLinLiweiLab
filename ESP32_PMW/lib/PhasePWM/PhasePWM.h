#ifndef PHASE_PWM_H
#define PHASE_PWM_H

#include <driver/mcpwm_prelude.h>
#include <driver/gpio.h>
#include <math.h>

#define MAX_CHANNELS 6 // ESP32 typically has 2 Groups x 3 Operators = 6 Channels

class PhasePWM {
public:
    /**
     * @brief Construct a new Phase PWM object
     * * @param channel_pins Array of GPIO pins for the channels
     * @param total_channels Number of channels to use (Max 6)
     * @param sync_pin GPIO used for synchronization.
     * - If ROLE_CLIENT: Connect this to the Main Node's Channel 0 Pin.
     * - If ROLE_MAIN (and channels > 3): Connect Channel 0 Pin to this Sync Pin physically to bridge Group 0/1.
     * @param pwm_freq_hz Initial Frequency
     * @param role Define if this node is the Main clock source or a Client
     */
    PhasePWM(const int* channel_pins, size_t total_channels, int sync_pin, int pwm_freq_hz, bool is_main = true);

    bool begin();
    
    /**
     * @brief Set the Phase Shift for a specific channel
     * @param user_channel_index Index 0 to N-1
     * @param degrees Phase shift in degrees (0-360)
     */
    void setPhase(int user_channel_index, float degrees);

    /**
     * @brief Set the Duty Cycle
     * @param user_channel_index Index 0 to N-1
     * @param percent Duty cycle 0.0 to 100.0
     */
    void setDuty(int user_channel_index, float percent);

    void setFrequency(int pwm_freq_hz);

private:
    uint32_t degreesToTicks(float degrees);
    uint32_t percentToTicks(float percent);

    const int* _channel_pins;
    size_t _total_channels;
    int _sync_pin;
    int _pwm_freq;
    bool _is_main;
    
    uint32_t _timer_resolution = 1000000; // 1 MHz resolution for high precision
    uint32_t _period_ticks = 0;

    // Handles
    mcpwm_timer_handle_t _timers[MAX_CHANNELS] = {NULL};
    mcpwm_oper_handle_t _operators[MAX_CHANNELS] = {NULL};
    mcpwm_cmpr_handle_t _comparators[MAX_CHANNELS] = {NULL};
    mcpwm_gen_handle_t _generators[MAX_CHANNELS] = {NULL};

    // Sync Sources
    mcpwm_sync_handle_t _sync_source_internal = NULL; // Internal timer event (Group 0 only)
    mcpwm_sync_handle_t _sync_source_external = NULL; // GPIO Sync (Works for Group 0 and 1)

    // State Storage for Frequency changes
    float _stored_duties[MAX_CHANNELS];
    float _stored_phases[MAX_CHANNELS];
};

#endif // PHASE_PWM_H