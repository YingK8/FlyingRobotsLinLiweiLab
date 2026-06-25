#include "PhaseSequencer.h"
#include <math.h>

PhaseSequencer::PhaseSequencer(PhaseController *phaseCtrl) {
  _phaseCtrl = phaseCtrl;
  _currentFrameIdx = 0;
  _taskStartTimeUs = 0;
  _taskFrameOffsetUs = 0;
  _taskStepUs = 1000;
  _initialFreqHz = 0.0f;
  _currentFreqHz = 0.0f;
  for (int i = 0; i < 4; i++) {
    _initialDutyCycles[i] = 0.0f;
    _initialPhaseDegrees[i] = 0.0f;
    _currentDutyCycles[i] = 0.0f;
    _currentPhaseDegrees[i] = 0.0f;
    _currentCarrierDutyCycles[i] = NAN;
  }
}

void PhaseSequencer::reserve(size_t size) { _queue.reserve(size); }

void PhaseSequencer::addSequenceTask(SequenceTask task) {
  _queue.push_back(task);
}

// Updated builders to zero-initialize the new struct fields automatically
void PhaseSequencer::addDutyCycleTask(const float *dutyCycles,
                                      int numChannels) {
  SequenceTask task = {}; // Zero-initializes all fields
  task.type = TASK_SET_DUTY_CYCLES;
  for (int i = 0; i < numChannels; i++) {
    float value = dutyCycles[i];
    if (value < 0.0f)
      value = 0.0f;
    if (value > 100.0f)
      value = 100.0f;
    task.dutyCycles[i] = value;
  }
  _queue.push_back(task);
}

// Add a carrier duty cycle set task for all channels.
void PhaseSequencer::addCarrierDutyCycleTask(const float *carrierDutyCycles,
                                             int numChannels) {
  SequenceTask task = {}; // Zero-initializes all fields
  task.type = TASK_SET_CARRIER_DUTY_CYCLES;
  for (int i = 0; i < numChannels; i++) {
    float value = carrierDutyCycles[i];
    if (value < 0.0f)
      value = 0.0f;
    if (value > 100.0f)
      value = 100.0f;
    task.carrierDuties[i] = value;
  }
  _queue.push_back(task);
}

void PhaseSequencer::addPhaseTask(const float *phases, int numChannels) {
  SequenceTask task = {};
  task.type = TASK_SET_PHASES;
  for (int i = 0; i < numChannels; i++) {
    task.endPhases[i] = phases[i];
  }
  _queue.push_back(task);
}

void PhaseSequencer::addWaitTask(uint32_t durationMs) {
  SequenceTask task = {};
  task.type = TASK_WAIT;
  task.durationUs = (int64_t)durationMs * 1000LL;
  _queue.push_back(task);
}

void PhaseSequencer::addLinearRampTask(float startHz, float endHz,
                                       uint32_t durationMs) {
  if (durationMs == 0)
    durationMs = 1;
  SequenceTask task = {};
  task.type = TASK_RAMP_LINEAR;
  task.durationUs = (int64_t)durationMs * 1000LL;
  task.startFreq = startHz;
  task.endFreq = endHz;
  _queue.push_back(task);
}

void PhaseSequencer::addEaseRampTask(float startHz, float endHz,
                                     uint32_t durationMs) {
  if (durationMs == 0)
    durationMs = 1;
  SequenceTask task = {};
  task.type = TASK_RAMP_EASE;
  task.durationUs = (int64_t)durationMs * 1000LL;
  task.startFreq = startHz;
  task.endFreq = endHz;
  _queue.push_back(task);
}

void PhaseSequencer::addPhaseRampTask(const float *startPhases,
                                      const float *endPhases,
                                      uint32_t durationMs) {
  if (durationMs == 0)
    durationMs = 1;
  SequenceTask task = {};
  task.type = TASK_RAMP_PHASE;
  task.durationUs = (int64_t)durationMs * 1000LL;
  for (int i = 0; i < 4; i++) {
    task.startPhases[i] = startPhases[i];
    task.endPhases[i] = endPhases[i];
  }
  _queue.push_back(task);
}

void PhaseSequencer::addDisableTask() {
  SequenceTask task = {};
  task.type = TASK_DISABLE_OUTPUTS;
  _queue.push_back(task);
}

// SAFETY: disable the VNH5019. Pin the streaming state to 0 so any later
// applyCurrentState() keeps the bridge idle instead of re-driving it, then defer
// to PhaseController::disableOutputs() for the actual coast (PWM 0 + carrier 0)
// and the gate-level EN-pin disable (if an enablePin was configured there).
void PhaseSequencer::disableOutputs() {
  if (!_phaseCtrl)
    return;
  for (int i = 0; i < 4; i++) {
    _currentDutyCycles[i] = 0.0f;
    _currentCarrierDutyCycles[i] = 0.0f;
  }
  _phaseCtrl->disableOutputs();
}

float PhaseSequencer::easeInOut(float t) {
  if (t < 0.0f)
    t = 0.0f;
  if (t > 1.0f)
    t = 1.0f;
  return t * t * (3.0f - 2.0f * t);
}

void PhaseSequencer::resetStreamingState() {
  _currentFrameIdx = 0;
  _taskStartTimeUs = 0;
  _taskFrameOffsetUs = 0;
  _currentFreqHz = _initialFreqHz;

  for (int i = 0; i < 4; i++) {
    _currentDutyCycles[i] = _initialDutyCycles[i];
    _currentPhaseDegrees[i] = _initialPhaseDegrees[i];
    _currentCarrierDutyCycles[i] = NAN;
  }
}

void PhaseSequencer::applyCurrentState() {
  if (!_phaseCtrl)
    return;

  _phaseCtrl->setGlobalFrequency(_currentFreqHz);

  for (int i = 0; i < 4; i++) {
    _phaseCtrl->setDutyCycle(i, _currentDutyCycles[i]);
    _phaseCtrl->setPhase(i, _currentPhaseDegrees[i]);

    if (!isnan(_currentCarrierDutyCycles[i])) {
      _phaseCtrl->setCarrierDutyCycle(i, _currentCarrierDutyCycles[i]);
    }
  }
}

void PhaseSequencer::compile(uint32_t resolutionMs, float initialFreq,
                             const float *initialDuty,
                             const float *initialPhase) {
  _initialFreqHz = initialFreq;
  _taskStepUs = (int64_t)resolutionMs * 1000LL;
  if (_taskStepUs <= 0)
    _taskStepUs = 1000;

  for (int i = 0; i < 4; i++) {
    _initialDutyCycles[i] = initialDuty ? initialDuty[i] : 0.0f;
    _initialPhaseDegrees[i] = initialPhase ? initialPhase[i] : 0.0f;
  }

  resetStreamingState();
}

void PhaseSequencer::start() {
  _currentFrameIdx = 0;
  _taskStartTimeUs = esp_timer_get_time();
  _taskFrameOffsetUs = 0;
  _currentFreqHz = _initialFreqHz;

  for (int i = 0; i < 4; i++) {
    _currentDutyCycles[i] = _initialDutyCycles[i];
    _currentPhaseDegrees[i] = _initialPhaseDegrees[i];
    _currentCarrierDutyCycles[i] = NAN;
  }
}

bool PhaseSequencer::isDone() const {
  return _currentFrameIdx >= _queue.size();
}

// =========================================================
// HIGH-SPEED HOT LOOP: No math, just pointer lookups
// =========================================================
void PhaseSequencer::run() {
  if (_queue.empty() || _currentFrameIdx >= _queue.size())
    return;

  int64_t nowUs = esp_timer_get_time();

  while (_currentFrameIdx < _queue.size()) {
    SequenceTask &task = _queue[_currentFrameIdx];
    int64_t elapsedUs = nowUs - _taskStartTimeUs;
    if (elapsedUs < 0)
      elapsedUs = 0;

    if (task.type == TASK_WAIT) {
      if (elapsedUs < task.durationUs)
        return;
      _currentFrameIdx++;
      _taskStartTimeUs = nowUs;
      _taskFrameOffsetUs = 0;
      continue;
    }

    if (task.type == TASK_SET_DUTY_CYCLES) {
      for (int i = 0; i < 4; i++) {
        _currentDutyCycles[i] = task.dutyCycles[i];
      }
      applyCurrentState();
      _currentFrameIdx++;
      _taskStartTimeUs = nowUs;
      _taskFrameOffsetUs = 0;
      continue;
    }

    if (task.type == TASK_SET_CARRIER_DUTY_CYCLES) {
      for (int i = 0; i < 4; i++) {
        _currentCarrierDutyCycles[i] = task.carrierDuties[i];
      }
      applyCurrentState();
      _currentFrameIdx++;
      _taskStartTimeUs = nowUs;
      _taskFrameOffsetUs = 0;
      continue;
    }

    if (task.type == TASK_SET_PHASES) {
      for (int i = 0; i < 4; i++) {
        _currentPhaseDegrees[i] = task.endPhases[i];
      }
      applyCurrentState();
      _currentFrameIdx++;
      _taskStartTimeUs = nowUs;
      _taskFrameOffsetUs = 0;
      continue;
    }

    if (task.type == TASK_DISABLE_OUTPUTS) {
      disableOutputs();  // coast (+ EN-pin disable if configured on controller)
      _currentFrameIdx++;
      _taskStartTimeUs = nowUs;
      _taskFrameOffsetUs = 0;
      continue;
    }

    if (task.type == TASK_TRAJECTORY_POINT) {
      _currentFreqHz = task.startFreq;
      for (int i = 0; i < 4; i++) {
        _currentDutyCycles[i] = task.dutyCycles[i];
        _currentPhaseDegrees[i] = task.startPhases[i];
        _currentCarrierDutyCycles[i] = task.carrierDuties[i];
      }
      applyCurrentState();
      _currentFrameIdx++;
      _taskStartTimeUs = nowUs;
      _taskFrameOffsetUs = 0;
      continue;
    }

    if (task.type == TASK_RAMP_LINEAR || task.type == TASK_RAMP_EASE ||
        task.type == TASK_RAMP_PHASE) {
      if (task.durationUs <= 0) {
        if (task.type == TASK_RAMP_PHASE) {
          for (int i = 0; i < 4; i++) {
            _currentPhaseDegrees[i] = task.endPhases[i];
          }
        } else {
          _currentFreqHz = task.endFreq;
        }
        applyCurrentState();
        _currentFrameIdx++;
        _taskStartTimeUs = nowUs;
        _taskFrameOffsetUs = 0;
        continue;
      }

      int64_t sampleOffsetUs = _taskFrameOffsetUs;
      while (sampleOffsetUs <= elapsedUs && sampleOffsetUs <= task.durationUs) {
        float t = (float)sampleOffsetUs / (float)task.durationUs;
        if (task.type == TASK_RAMP_EASE || task.type == TASK_RAMP_PHASE)
          t = easeInOut(t);

        if (task.type == TASK_RAMP_PHASE) {
          for (int i = 0; i < 4; i++) {
            _currentPhaseDegrees[i] = task.startPhases[i] +
                                       t * (task.endPhases[i] - task.startPhases[i]);
          }
        } else {
          _currentFreqHz = task.startFreq + t * (task.endFreq - task.startFreq);
        }

        applyCurrentState();

        if (sampleOffsetUs >= task.durationUs)
          break;
        sampleOffsetUs += _taskStepUs;
        _taskFrameOffsetUs = sampleOffsetUs;
      }

      if (elapsedUs < task.durationUs)
        return;

      if (task.type == TASK_RAMP_PHASE) {
        for (int i = 0; i < 4; i++) {
          _currentPhaseDegrees[i] = task.endPhases[i];
        }
      } else {
        _currentFreqHz = task.endFreq;
      }
      applyCurrentState();

      _currentFrameIdx++;
      _taskStartTimeUs = nowUs;
      _taskFrameOffsetUs = 0;
      continue;
    }

    _currentFrameIdx++;
    _taskStartTimeUs = nowUs;
    _taskFrameOffsetUs = 0;
  }
}