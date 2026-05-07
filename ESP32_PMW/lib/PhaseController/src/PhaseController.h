#pragma once

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_timer.h"
#include <Arduino.h>

// Configuration Defines
// (Ensure these match your project settings or are defined here)
// #ifndef USE_SYNC
// #define USE_SYNC 1
// #endif

// Set to 1 for Master (Output Sync), 0 for Client (Input Sync)
// #ifndef SYNC_AS_SERVER
// #define SYNC_AS_SERVER 0
// #endif

#define FREQ_FILTER_SIZE 5

// #define SYNC_LATENCY_US 50 // Tuning value for latency compensation

struct PhaseParams {
  unsigned long startUs;
  unsigned long endUs;
  bool wraps;
};

class PhaseController {
public:
  /**
   * @brief Construct a PhaseController for multi-channel PWM output.
   * @param pins Array of GPIO pins for PWM output.
   * @param phaseOffsetsDegrees Array of initial phase offsets (degrees).
   * @param dutyCycles Array of initial duty cycles (%).
   * @param numChannels Number of PWM channels.
   */
  PhaseController(const gpio_num_t *pins, const float *phaseOffsetsDegrees,
                  const float *dutyCycles, int numChannels);
  ~PhaseController();

  /**
   * @brief Initialize hardware and start PWM generation at the given frequency.
   * @param initialFreqHz Initial frequency in Hz.
   */
  void begin(float initialFreqHz=0.0f);

  /**
   * @brief Main loop task for drift compensation. Call regularly in loop().
   */
  void run();

  // Configuration
  /**
   * @brief Set the global frequency for all PWM channels.
   * @param newHz Frequency in Hz.
   */
  void setGlobalFrequency(float newHz);
  /**
   * @brief Set the frequency for a specific channel.
   * @param channel Channel index.
   * @param newHz Frequency in Hz.
   */
  void setFrequency(int channel, float newHz);
  /**
   * @brief Set the duty cycle for a specific channel.
   * @param channel Channel index.
   * @param dutyPercent Duty cycle percentage (0-100).
   */
  void setDutyCycle(int channel, float dutyPercent);
  /**
   * @brief Set the phase for a specific channel.
   * @param channel Channel index.
   * @param degrees Phase in degrees (0-360).
   */
  void setPhase(int channel, float degrees);

  // Getters
  /**
   * @brief Get the global frequency.
   * @return Frequency in Hz.
   */
  float getFrequency() const;
  /**
   * @brief Get the phase for a specific channel.
   * @param channel Channel index.
   * @return Phase in degrees.
   */
  float getPhase(int channel) const;
  /**
   * @brief Get the duty cycle for a specific channel.
   * @param channel Channel index.
   * @return Duty cycle percentage.
   */
  float getDutyCycle(int channel) const;

  // Sync Configuration
  /**
   * @brief Enable synchronization using a sync pin.
   * @param syncPin GPIO pin for sync signal.
   */
  void enableSync(gpio_num_t syncPin);

  // Carrier PWM Configuration (multi-channel)
  /**
   * @brief Initialize carrier PWM for all channels.
   * @param pins Array of GPIO pins for carrier PWM.
   * @param freqHz Carrier frequency in Hz.
   * @param dutyPercents Array of duty cycles for each channel.
   */
  void initCarrierPWM(const gpio_num_t *pins, float freqHz,
                      const float *dutyPercents);
  /**
   * @brief Set the carrier PWM duty cycle for a specific channel.
   * @param channel Channel index.
   * @param dutyPercent Duty cycle percentage (0-100).
   */
  void setCarrierDutyCycle(int channel, float dutyPercent);

private:
  // Minimum on/off period constraint: 0.0 ms
  const float MIN_ON_OFF_MS = 0.0f;

  // Internal methods
  static void IRAM_ATTR _timerCallback(void *arg);
  static void IRAM_ATTR _onSyncInterrupt();
  void updatePhaseParams(int channel);

  // Hardware
  int _numChannels;
  gpio_num_t *_pins;
  esp_timer_handle_t _periodicTimer;
  gpio_num_t _syncPin;

  // Carrier PWM (multi-channel)
  gpio_num_t *_carrierPinsArray = nullptr;
  float _carrierFreqHz;
  float *_carrierDutyCyclePct = nullptr;
  ledc_mode_t _carrierSpeedMode = LEDC_LOW_SPEED_MODE;
  ledc_timer_t _carrierTimer = LEDC_TIMER_0;
  uint8_t _carrierDutyResolutionBits = 10;

  // State Arrays
  float *_phaseOffsetsPct;
  float *_dutyCycles;
  PhaseParams *_params;
  float _globalFreqHz; // New global frequency variable

  // Sync State
  int64_t _lastSyncTimeUs;
  int64_t _averagedPeriodUs;
  int64_t _periodBuffer[FREQ_FILTER_SIZE];
  int _filterIdx;

  // ISR variables
  static PhaseController *_isrInstance;
  int64_t _lastIsrTimeUs;
  bool _firstSyncReceived;

  // Concurrency
  portMUX_TYPE _spinlock = portMUX_INITIALIZER_UNLOCKED;
};
