#pragma once

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_timer.h"
#include <Arduino.h>

#include "CurrentBalanceController.h" // folded-in current-balance PI (opt-in)
#include "current_sense.h"            // folded-in VNH5019 CS reader (opt-in)

#define FREQ_FILTER_SIZE 5

struct PhaseParams {
  unsigned long startUs;
  unsigned long endUs;
  bool wraps;
};

class PwmController {
public:
  // Per-channel arrays of length numChannels: pins, phase offsets (deg),
  // duty cycles (%).
  PwmController(const gpio_num_t *pins, const float *phaseOffsetsDegrees,
                  const float *dutyCycles, int numChannels);
  ~PwmController();

  void begin(float initialFreqHz = 0.0f); ///< Init hardware; default (0) starts in DC (stationary, non-rotating) mode.
  void run();                             ///< Drift compensation; call every loop().

  // Configuration
  void setGlobalFrequency(float newHz);              ///< All channels, Hz.
  void setFrequency(int channel, float newHz);       ///< One channel, Hz.
  void setDutyCycle(int channel, float dutyPercent); ///< 0-100%.
  void setPhase(int channel, float degrees);         ///< 0-360 deg.

  // Getters
  float getFrequency() const;            ///< Global frequency (Hz); 0 in DC mode.
  bool isDC() const { return _dcMode; }  ///< True when the field is held static (freq <= 0).
  float getPhase(int channel) const;     ///< Phase (deg).
  float getDutyCycle(int channel) const; ///< Duty (%).

  void enableSync(gpio_num_t syncPin); ///< Sync PWM to an external pulse on syncPin.

  // Carrier PWM (multi-channel). pins[]/dutyPercents[] per channel; freqHz shared.
  void initCarrierPWM(const gpio_num_t *pins, float freqHz,
                      const float *dutyPercents);
  /// Set carrier duty (0-100%). Becomes the per-channel ceiling when balance is on.
  void setCarrierDutyCycle(int channel, float dutyPercent);
  /// Carrier duty (%), or 0 if carrier PWM uninitialized / channel out of range.
  float getCarrierDutyCycle(int channel) const;

  // ==================== Current sense + PI balance (opt-in) ====================
  // Folded into the controller: a main opts in and run() does the work. Both are
  // OFF by default, so an experiment that never calls these stays pure open-loop
  // passthrough.

  /**
   * @brief Enable current sensing (+ optional overcurrent latch); run() then
   *        paces the ADC and updates measuredCurrents(). Seeds the zero offset
   *        immediately: call at boot with coils confirmed OFF (forceAllGatesLow()
   *        + carriers at 0). Usable for telemetry/safety without the balance loop.
   * @param adcPins           VNH5019 CS ADC pins (constants.h::ADC_PINS).
   * @param sensPerVolt       per-board CS calibration, A/V (4 channels).
   * @param overcurrentTripA  per-channel latch level (A); 0 disables.
   */
  void enableCurrentSense(const gpio_num_t *adcPins, const float *sensPerVolt,
                          float overcurrentTripA = 0.0f);

  /**
   * @brief Enable the current-balance PI loop (call enableCurrentSense first).
   *        setCarrierDutyCycle(i, pct) then means the per-channel carrier CEILING;
   *        run() drives actual duty beneath it so the four currents converge
   *        (equal, or ratio'd when the schedule steps channels down for tilt).
   *        Leave off for characterization sweeps (carriers pass through verbatim).
   * @param cfg        balance tuning (defaults = converged KP/KI/KD).
   * @param startDuty  duty every channel begins equal at (default 50%).
   */
  void enableCurrentBalance(const BalanceConfig &cfg = BalanceConfig(),
                            float startDuty = 50.0f);

  // Runtime balance tuning (the current_pid serial rig: kp=/ki=/kd=/ramp=).
  void setBalanceGains(float kp, float ki, float kd);
  
  void setBalanceRamp(float pctPerMs);

  /** @brief Latest filtered per-channel current (A), or nullptr if sensing is
   *  off. Valid for the lifetime of the controller. */
  const float *measuredCurrents() const;

  /** @brief The per-channel carrier ceiling the schedule last commanded while
   *  balance is active (what setCarrierDutyCycle stored), or the live carrier
   *  duty when balance is off. */
  float carrierCeiling(int channel) const;

  bool balanceActive() const { return _balance != nullptr; }

  bool currentSenseActive() const { return _sense != nullptr; }

  /** @brief True once an overcurrent trip has latched all carriers off. */
  bool overcurrentTripped() const { return _tripped; }

  /**
   * @brief Gracefully de-energize all coils: ramp every carrier duty down to 0
   *        over rampMs, then stop the periodic phase timer so all output halts.
   *        The controller/timer remain allocated (unlike the destructor).
   *        BLOCKING (uses delay()); for a non-blocking ramp use rampDownStep().
   * @param rampMs Duration of the linear ramp-down in milliseconds.
   */
  void shutdown(unsigned long rampMs = 2000);

  /**
   * @brief One monotonic ramp-down step for a non-blocking safety ramp. Reads
   *        each channel's CURRENT carrier duty and subtracts stepPct (clamped at
   *        0), then applies it. Because it subtracts from the live value instead
   *        of interpolating toward a pre-set target, the commanded duty can only
   *        ever decrease: no configuration can momentarily drive it upward.
   *        Call once per control tick until it returns true.
   * @param stepPct Percentage points to drop each call (must be > 0).
   * @return true once every channel has reached 0.
   */
  bool rampDownStep(float stepPct);

private:
  // Minimum on/off period constraint: 0.0 ms
  const float MIN_ON_OFF_MS = 0.0f;

  // Internal methods
  static void IRAM_ATTR _timerCallback(void *arg);
  static void IRAM_ATTR _onSyncInterrupt();
  void updatePhaseParams(int channel);
  // Actually write a carrier duty to the LEDC hardware (the body that
  // setCarrierDutyCycle used to be). setCarrierDutyCycle now routes through here
  // for passthrough, or stashes a ceiling for the balance loop to drive.
  void _writeCarrier(int channel, float dutyPercent);
  // Sense/balance work done inside run() when opted in.
  void _serviceCurrentLoop();

  // Current sense + PI balance (opt-in; both null unless enabled).
  CurrentSense *_sense = nullptr;
  CurrentBalanceController *_balance = nullptr;
  float _ceiling[4] = {NAN, NAN, NAN, NAN}; // per-channel carrier ceiling (balance on)
  float _balanceDuty[4] = {0, 0, 0, 0};     // last PI-computed duties (telemetry)
  float _startDuty = 50.0f;
  float _overcurrentTripA = 0.0f;           // 0 => trip disabled
  bool _tripped = false;
  unsigned long _lastSenseUs = 0;
  unsigned long _lastBalanceUs = 0;

  // Hardware
  int _numChannels;
  gpio_num_t *_pins;
  esp_timer_handle_t _periodicTimer;
  gpio_num_t _syncPin;

  // Carrier PWM (multi-channel)
  gpio_num_t *_carrierPinsArray = nullptr;
  float _carrierFreqHz;
  float *_carrierDutyCyclePct = nullptr;
  bool *_carrierLedcConfigured = nullptr;   // LEDC channel attached to the pin?
  uint32_t *_carrierLastDutyTicks = nullptr; // last duty written, in timer ticks
  ledc_mode_t _carrierSpeedMode = LEDC_LOW_SPEED_MODE;
  ledc_timer_t _carrierTimer = LEDC_TIMER_0;
  uint8_t _carrierDutyResolutionBits = 10;

  // State Arrays
  float *_phaseOffsetsPct;
  float *_dutyCycles;
  PhaseParams *_params;
  float _globalFreqHz; // New global frequency variable
  bool _dcMode = false; // true => field held static (no rotation); see setGlobalFrequency

  // Sync State
  int64_t _lastSyncTimeUs;
  int64_t _averagedPeriodUs;
  int64_t _periodBuffer[FREQ_FILTER_SIZE];
  int _filterIdx;

  // ISR variables
  static PwmController *_isrInstance;
  int64_t _lastIsrTimeUs;
  bool _firstSyncReceived;

  // Concurrency
  portMUX_TYPE _spinlock = portMUX_INITIALIZER_UNLOCKED;
};
