#pragma once
// Minimal cross-core shared state between core 0 (PwmActuator: ADC/PWM/state
// machine/safety) and core 1 (CurrentBalanceController: feedforward/LQID
// feedback/allocator, comm, telemetry). Two independent single-writer/
// single-reader regions, each behind its own portMUX_TYPE spinlock -- the
// same minimal idiom already used in lib/PWMController/src/PWMController.cpp
// for its own ISR-vs-caller synchronization. No queues/semaphores: payloads
// are a few floats, critical sections are copy-in/copy-out only.
//
// No staleness watchdog by design: core 0 just applies whatever's currently
// in the duty region, indifferent to whether it's fresh or stale -- safety
// comes from core 0's own independent overcurrent trip and self-contained
// bounded state machine, not from detecting a slow/dead core 1.

#include "constants.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

class SharedMemory {
public:
  SharedMemory();

  // core0 -> core1 ("log memory")
  void publishMeasurement(const float i_meas[NUM_CHANNELS], int phase, float freq);
  void readMeasurement(float i_meas_out[NUM_CHANNELS], int *phase_out, float *freq_out) const;

  // core1 -> core0 ("command memory")
  void publishDuty(const float duty[NUM_CHANNELS]);
  void readDuty(float duty_out[NUM_CHANNELS]) const;

  // core1 -> core0, rare: direction-change request. core0 (PwmActuator)
  // only honors this when its own authoritative phase is ARMING/STOPPED --
  // see PwmActuator::applyDirectionRequestIfSafe().
  void requestDirection(bool ccw);
  bool consumeDirectionRequest(bool *ccw_out); // core0 side only

private:
  mutable portMUX_TYPE _logMux;
  float _i_meas[NUM_CHANNELS];
  int _phase;
  float _freq;

  mutable portMUX_TYPE _cmdMux;
  float _duty[NUM_CHANNELS];

  mutable portMUX_TYPE _dirMux;
  bool _dirRequestPending;
  bool _dirRequestCcw;
};
