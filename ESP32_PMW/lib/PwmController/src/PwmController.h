#pragma once

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_timer.h"
#include <Arduino.h>

#include "CurrentBalanceController.h"
#include "current_sense.h"

#define FREQ_FILTER_SIZE 5

// bounds the callback's on-stack snapshot: raising it costs ISR stack.
#define PWMC_MAX_CHANNELS 8

struct PhaseParams {
  unsigned long startUs;
  unsigned long endUs;
  bool wraps;
};

class PwmController {
public:
  PwmController(const gpio_num_t *pins, const float *phaseOffsetsDegrees,
                  const float *dutyCycles, int numChannels);
  ~PwmController();

  void begin(float initialFreqHz = 0.0f);
  void run();

  void setGlobalFrequency(float newHz);
  void setDutyCycle(int channel, float dutyPercent);
  void setPhase(int channel, float degrees);
  void setPhases(const float *degrees);

  void setPhaseTrim(const float *f0Hz, const float *q);

  bool phaseTrimActive() const { return _trimOn; }

  float phaseTrimDeg(int channel) const;

  float getFrequency() const;


  void initCarrierPWM(const gpio_num_t *pins, float freqHz,
                      const float *dutyPercents);
  void setCarrierDutyCycle(int channel, float dutyPercent);
  float getCarrierDutyCycle(int channel) const;


  void enableCurrentSense(const gpio_num_t *adcPins, const float *sensPerVolt,
                          float overcurrentTripA = 0.0f);

  void enableCurrentBalance(const BalanceConfig &cfg = BalanceConfig(),
                            float startDuty = 50.0f);



  const float *measuredCurrents() const;


  bool balanceActive() const { return _balance != nullptr; }


  bool overcurrentTripped() const { return _tripped; }


  bool rampDownStep(float stepPct);

private:
  // 0.0 ms: no minimum on/off guard is applied.
  const float MIN_ON_OFF_MS = 0.0f;

  static void IRAM_ATTR _timerCallback(void *arg);
  static void IRAM_ATTR _onSyncInterrupt();
  void updatePhaseParams(int channel);
  void _applyPhase(int channel);
  void _recomputeTrim();
  void _writeCarrier(int channel, float dutyPercent);
  void _serviceCurrentLoop();

  CurrentSense *_sense = nullptr;
  CurrentBalanceController *_balance = nullptr;
  float _ceiling[4] = {NAN, NAN, NAN, NAN};
  float _balanceDuty[4] = {0, 0, 0, 0};
  float _startDuty = 50.0f;
  float _overcurrentTripA = 0.0f;
  bool _tripped = false;
  unsigned long _lastSenseUs = 0;
  unsigned long _lastBalanceUs = 0;

  int _numChannels;
  // commanded phase, kept apart so a trim never accumulates into the base.
  gpio_num_t *_pins;
  esp_timer_handle_t _periodicTimer;
  gpio_num_t _syncPin;

  gpio_num_t *_carrierPinsArray = nullptr;
  float _carrierFreqHz;
  float *_carrierDutyCyclePct = nullptr;
  bool *_carrierLedcConfigured = nullptr;
  uint32_t *_carrierLastDutyTicks = nullptr;
  ledc_mode_t _carrierSpeedMode = LEDC_LOW_SPEED_MODE;
  ledc_timer_t _carrierTimer = LEDC_TIMER_0;
  uint8_t _carrierDutyResolutionBits = 10;

  float *_phaseOffsetsPct;
  float *_basePhaseDeg;
  // read inside the spinlock: recompute the trim only outside it.
  float _trimF0Hz[PWMC_MAX_CHANNELS] = {0};
  float _trimQ[PWMC_MAX_CHANNELS] = {0};
  float _trimCacheDeg[PWMC_MAX_CHANNELS] = {0};
  bool _trimOn = false;
  static float _wrapPct(float degrees);
  float *_dutyCycles;
  PhaseParams *_params;
  float _globalFreqHz;
  bool _dcMode = false;

  int64_t _lastSyncTimeUs;
  int64_t _averagedPeriodUs;
  int64_t _periodBuffer[FREQ_FILTER_SIZE];
  int _filterIdx;

  static PwmController *_isrInstance;
  int64_t _lastIsrTimeUs;
  bool _firstSyncReceived;

  portMUX_TYPE _spinlock = portMUX_INITIALIZER_UNLOCKED;
};
