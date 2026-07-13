#include "PwmActuator.h"
#include "state_space_constants.h"
#include <Arduino.h>

static const float PHASES_CW[NUM_CHANNELS] = {270.0, 90.0, 180.0, 0.0};
static const float PHASES_CCW[NUM_CHANNELS] = {90.0, 270.0, 180.0, 0.0};
static const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
static const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {0, 0, 0, 0};
static const float SENS[NUM_CHANNELS] = {15.26, 15.28, 15.57, 15.34};

static const float START_FREQ = 1.0f;
static const float END_FREQ = 210.0f;
static const unsigned long RAMP_DURATION_MS = 20000;
static const unsigned long ARM_MS = 3000;
static const unsigned long HOLD_MS = 5000;
static const unsigned long RAMP_TICK_MS = 20;
static const float END_STEP_PCT = 2.0f;
static const unsigned long ADC_SAMPLE_MS = 1;

PwmActuator::PwmActuator(SharedMemory *shared)
    : _shared(shared), _controller(nullptr), _seq(nullptr), _currentSense(ADC_PINS, SENS, NUM_CHANNELS),
      _directionIsCcw(true), _phase(PHASE_ARMING), _phase_start(0), _last_ramp_ms(0), _last_adc_us(0) {}

void PwmActuator::reinitController(bool ccw) {
  const float *phases = ccw ? PHASES_CCW : PHASES_CW;
  _directionIsCcw = ccw;

  if (_controller) delete _controller;
  _controller = new PWMController(PWM_PINS, phases, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  _controller->begin(START_FREQ);
  _controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  allCoilsOff();

  if (_seq) delete _seq;
  _seq = new PWMSequencer(_controller);
  _seq->addRampTask(START_FREQ, END_FREQ, RAMP_DURATION_MS, TaskType::PWM_FREQ, TaskMode::EASE);
  _seq->compile(25, 1.0f, INITIAL_DUTY_CYCLES, phases);
}

void PwmActuator::allCoilsOff() {
  for (int i = 0; i < NUM_CHANNELS; i++) _controller->setCarrierDutyCycle(i, 0.0f);
}

void PwmActuator::applyCommandedDuty() {
  float duty[NUM_CHANNELS];
  _shared->readDuty(duty);
  for (int i = 0; i < NUM_CHANNELS; i++) _controller->setCarrierDutyCycle(i, duty[i]);
}

void PwmActuator::begin() {
  _currentSense.seed();
  reinitController(_directionIsCcw);
  _phase = PHASE_ARMING;
  _phase_start = millis();
  _last_adc_us = micros();
}

void PwmActuator::applyDirectionRequestIfSafe() {
  // Only core 0 ever mutates _controller/_seq, avoiding any race with its
  // own concurrent use of them. Only honored while genuinely safe to
  // reconstruct the controller -- double-guards core 1's own gating (which
  // necessarily checks a slightly-stale copy of phase).
  if (_phase != PHASE_ARMING && _phase != PHASE_STOPPED) return;
  bool requestedCcw;
  if (_shared->consumeDirectionRequest(&requestedCcw)) reinitController(requestedCcw);
}

bool PwmActuator::checkOvercurrentTrip() {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    if (_currentSense.i_meas[i] > I_MAX_A) {
      allCoilsOff();
      digitalWrite(LED_PIN, LOW);
      _phase = PHASE_STOPPED; // immediate hard latch-off, skip the gentle ENDING ramp -- this is the emergency backstop
      return true;
    }
  }
  return false;
}

void PwmActuator::run() {
  unsigned long now = millis();

  unsigned long now_us = micros();
  float dt_since_adc_ms = (float)(now_us - _last_adc_us) / 1000.0f;
  if (dt_since_adc_ms >= (float)ADC_SAMPLE_MS) {
    _currentSense.update(dt_since_adc_ms);
    _last_adc_us = now_us;
  }

  _controller->step();
  if (_phase == PHASE_RAMP_UP || _phase == PHASE_HOLD) _seq->step();

  // Independent hard overcurrent trip -- checked every tick regardless of
  // core 1's health, latches off immediately if tripped.
  if (_phase != PHASE_STOPPED && checkOvercurrentTrip()) {
    _shared->publishMeasurement(_currentSense.i_meas, _phase, _controller->getFrequency());
    return;
  }

  switch (_phase) {
    case PHASE_ARMING:
      allCoilsOff();
      digitalWrite(LED_PIN, (now / 150) & 1);
      _currentSense.recalibrateZero();
      applyDirectionRequestIfSafe();
      if (now - _phase_start >= ARM_MS) {
        _seq->start();
        _phase = PHASE_RAMP_UP;
        _phase_start = now;
      }
      break;

    case PHASE_RAMP_UP:
      applyCommandedDuty();
      if (_controller->getFrequency() >= END_FREQ - 0.5f) {
        _phase = PHASE_HOLD;
        _phase_start = now;
      }
      break;

    case PHASE_HOLD:
      applyCommandedDuty();
      if (now - _phase_start >= HOLD_MS) {
        _phase = PHASE_ENDING;
        _last_ramp_ms = now;
      }
      break;

    case PHASE_ENDING:
      if (now - _last_ramp_ms >= RAMP_TICK_MS) {
        _last_ramp_ms = now;
        if (_controller->rampDownStep(END_STEP_PCT)) {
          digitalWrite(LED_PIN, LOW);
          _phase = PHASE_STOPPED;
        }
      }
      break;

    case PHASE_STOPPED:
      allCoilsOff();
      digitalWrite(LED_PIN, (now / 1000) & 1);
      applyDirectionRequestIfSafe();
      break;
  }

  _shared->publishMeasurement(_currentSense.i_meas, _phase, _controller->getFrequency());
}
