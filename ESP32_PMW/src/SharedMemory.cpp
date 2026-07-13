#include "SharedMemory.h"

SharedMemory::SharedMemory()
    : _logMux(portMUX_INITIALIZER_UNLOCKED), _phase(0), _freq(0.0f),
      _cmdMux(portMUX_INITIALIZER_UNLOCKED),
      _dirMux(portMUX_INITIALIZER_UNLOCKED), _dirRequestPending(false), _dirRequestCcw(false) {
  for (int i = 0; i < NUM_CHANNELS; i++) { _i_meas[i] = 0.0f; _duty[i] = 0.0f; }
}

void SharedMemory::publishMeasurement(const float i_meas[NUM_CHANNELS], int phase, float freq) {
  taskENTER_CRITICAL(&_logMux);
  for (int i = 0; i < NUM_CHANNELS; i++) _i_meas[i] = i_meas[i];
  _phase = phase;
  _freq = freq;
  taskEXIT_CRITICAL(&_logMux);
}

void SharedMemory::readMeasurement(float i_meas_out[NUM_CHANNELS], int *phase_out, float *freq_out) const {
  taskENTER_CRITICAL(&_logMux);
  for (int i = 0; i < NUM_CHANNELS; i++) i_meas_out[i] = _i_meas[i];
  *phase_out = _phase;
  *freq_out = _freq;
  taskEXIT_CRITICAL(&_logMux);
}

void SharedMemory::publishDuty(const float duty[NUM_CHANNELS]) {
  taskENTER_CRITICAL(&_cmdMux);
  for (int i = 0; i < NUM_CHANNELS; i++) _duty[i] = duty[i];
  taskEXIT_CRITICAL(&_cmdMux);
}

void SharedMemory::readDuty(float duty_out[NUM_CHANNELS]) const {
  taskENTER_CRITICAL(&_cmdMux);
  for (int i = 0; i < NUM_CHANNELS; i++) duty_out[i] = _duty[i];
  taskEXIT_CRITICAL(&_cmdMux);
}

void SharedMemory::requestDirection(bool ccw) {
  taskENTER_CRITICAL(&_dirMux);
  _dirRequestPending = true;
  _dirRequestCcw = ccw;
  taskEXIT_CRITICAL(&_dirMux);
}

bool SharedMemory::consumeDirectionRequest(bool *ccw_out) {
  bool had_request;
  taskENTER_CRITICAL(&_dirMux);
  had_request = _dirRequestPending;
  if (had_request) {
    *ccw_out = _dirRequestCcw;
    _dirRequestPending = false;
  }
  taskEXIT_CRITICAL(&_dirMux);
  return had_request;
}
