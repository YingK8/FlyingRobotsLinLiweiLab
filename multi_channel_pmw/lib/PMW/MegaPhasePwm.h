#ifndef MEGA_PHASE_PWM_H
#define MEGA_PHASE_PWM_H

#include <Arduino.h>

/**
 * @class MegaPhasePwm
 * @brief A driver for 6-channel, phase-shifted PWM on the ATmega2560.
 *
 * This class uses Timers 1, 3, and 4 to generate 6 independent PWM
 * channels with adjustable phase shifts between three timer banks.
 *
 * - Timer 1 (Bank 0): Channels 0 (Pin 11), 1 (Pin 12)
 * - Timer 3 (Bank 1): Channels 2 (Pin 5), 3 (Pin 2)
 * - Timer 4 (Bank 2): Channels 4 (Pin 6), 5 (Pin 7)
 *
 * Frequency, phase, and duty can be adjusted dynamically without stopping
 * the waveforms, unless a frequency change is large enough to
 * require a new prescaler.
 */
class MegaPhasePwm {
public:
    /**
     * @brief Constructor.
     */
    MegaPhasePwm();

    /**
     * @brief Initializes the timers, sets up ISRs, and starts the PWM.
     * @param initialFrequency The starting PWM frequency in Hz.
     */
    void begin(float initialFrequency = 300.0);

    /**
     * @brief Sets the master frequency for all 6 channels.
     * This will recalculate all timer settings and duty cycles.
     * This update is smooth (glitch-free) as long as the
     * prescaler doesn't need to change.
     * @param freq The desired frequency in Hz.
     * @return The actual frequency that was set (due to prescaler/timer resolution).
     */
    float setFrequency(float freq);

    /**
     * @brief Sets the duty cycle for an individual PWM channel.
     * This update is smooth and glitch-free.
     * @param channel The channel to modify (0-5).
     * @param duty The duty cycle as a float (0.0 for 0% to 1.0 for 100%).
     */
    void setDutyCycle(uint8_t channel, float duty);

    /**
     * @brief Sets the phase shift for a timer bank.
     * This update is smooth and glitch-free.
     * Timer 1 (Bank 0) is the reference (0 degrees).
     * @param timerBank The bank to shift: 0 (Timer 1), 1 (Timer 3), or 2 (Timer 4).
     * @param degrees The phase shift in degrees (0.0 - 360.0).
     */
    void setPhase(uint8_t timerBank, float degrees);

    void setPhaseMicroStep(uint8_t timerBank, float degrees, uint8_t microSteps = 10);
    float getPhaseResolution() const;
    float getFractionalPhase(uint8_t timerBank) const;

private:
    void stopTimers();
    void startTimersSynchronously();
    void setupPins();
    uint8_t calculatePrescaler(float freq, uint32_t& icrVal);
    void applyPhase(uint8_t timerBank, float degrees, bool keepFractional);

    // Member variables
    uint16_t m_icr;          // Master ICR value (TOP) for T1, T3, T4
    uint8_t m_prescalerBits; // TCCRxB bits for prescaler (e.g., 0x01, 0x02)
    float m_duty[6];       // Stored duty cycles for channels 0-5
    float m_phase[3];      // Stored phase in degrees for T1, T3, T4
    float m_phaseAccumulator[3] = {0};
};

#endif // MEGA_PHASE_PWM_H