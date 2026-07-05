#include "PhaseSequencer.h"
#include <math.h>

static float clampDuty(float v) {
  if (v < 0.0f)
    return 0.0f;
  if (v > 100.0f)
    return 100.0f;
  return v;
}

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

void PhaseSequencer::addWaitTask(uint32_t durationMs) {
  SequenceTask task = {};
  task.type = TaskType::WAIT;
  task.durationUs = (int64_t)durationMs * 1000LL;
  _queue.push_back(task);
}

// Global ramp
void PhaseSequencer::addRampTask(float start, float end, uint32_t durationMs,
                                 TaskType type, TaskMode ramp_mode) {
  const float starts[4] = {start, start, start, start};
  const float ends[4] = {end, end, end, end};
  if (type == TaskType::PWM_FREQ) {
    SequenceTask task = {};
    task.type = type;
    task.mode = ramp_mode;
    task.durationUs = (int64_t)durationMs * 1000LL;
    task.startFreq = start;
    task.endFreq = end;
    _queue.push_back(task);
    return;
  }
  addRampTask(starts, ends, 4, durationMs, type, ramp_mode);
}

// Per-channel ramp
void PhaseSequencer::addRampTask(const float *starts, const float *ends,
                                 int numChannels, uint32_t durationMs,
                                 TaskType type, TaskMode ramp_mode) {
  SequenceTask task = {};
  task.type = type;
  task.mode = ramp_mode;
  task.durationUs = (int64_t)durationMs * 1000LL;

  if (type == TaskType::PWM_FREQ) {
    // Frequency is global; only channel 0 is meaningful.
    task.startFreq = starts[0];
    task.endFreq = ends[0];
    _queue.push_back(task);
    return;
  }

  float *start_traj = nullptr;
  float *end_traj = nullptr;
  bool clamp = false;

  switch (type) {
  case TaskType::CARRIER_DUTY:
    start_traj = task.startCarriers;
    end_traj = task.endCarriers;
    clamp = true;
    break;
  case TaskType::PWM_DUTY:
    start_traj = task.startDuties;
    end_traj = task.endDuties;
    clamp = true;
    break;
  case TaskType::PWM_PHASE:
    start_traj = task.startPhases;
    end_traj = task.endPhases;
    break;
  default:
    return; // you can only ramp frequency, duty, carrier duty, or phase
  }

  for (int i = 0; i < 4; i++) {
    if (i < numChannels && !isnan(starts[i])) {
      float s = starts[i];
      float e = ends[i];
      if (clamp) {
        s = clampDuty(s);
        e = clampDuty(e);
      }
      start_traj[i] = s;
      end_traj[i] = e;
    } else {
      start_traj[i] = NAN; // if channel is not selected (NAN), skip
      end_traj[i] = NAN;
    }
  }
  _queue.push_back(task);
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

  // Push the initial state to the hardware immediately, so the configured
  // frequency/duty/phase are driven even before (or without) any queued task.
  applyCurrentState();
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

    if (task.type == TaskType::WAIT) {
      if (elapsedUs < task.durationUs)
        return;
      _currentFrameIdx++;
      _taskStartTimeUs = nowUs;
      _taskFrameOffsetUs = 0;
      continue;
    }

    if (task.type == TaskType::TRAJECTORY_POINT) {
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

    if (task.type == TaskType::PWM_FREQ || task.type == TaskType::PWM_DUTY ||
        task.type == TaskType::PWM_PHASE || task.type == TaskType::CARRIER_DUTY) {
      // Interpolate the quantity selected by task.type at fraction t. A
      // zero-duration task is an instant set (handled below via t = 1).
      auto applyRampAt = [&](float t) {
        switch (task.type) {
        case TaskType::CARRIER_DUTY:
          for (int i = 0; i < 4; i++)
            if (!isnan(task.startCarriers[i]))
              _currentCarrierDutyCycles[i] =
                  task.startCarriers[i] +
                  t * (task.endCarriers[i] - task.startCarriers[i]);
          break;
        case TaskType::PWM_DUTY:
          for (int i = 0; i < 4; i++)
            if (!isnan(task.startDuties[i]))
              _currentDutyCycles[i] =
                  task.startDuties[i] +
                  t * (task.endDuties[i] - task.startDuties[i]);
          break;
        case TaskType::PWM_PHASE:
          for (int i = 0; i < 4; i++)
            if (!isnan(task.startPhases[i]))
              _currentPhaseDegrees[i] =
                  task.startPhases[i] +
                  t * (task.endPhases[i] - task.startPhases[i]);
          break;
        case TaskType::PWM_FREQ:
        default:
          _currentFreqHz = task.startFreq + t * (task.endFreq - task.startFreq);
          break;
        }
      };

      if (task.durationUs <= 0) {
        applyRampAt(1.0f);
        applyCurrentState();
        _currentFrameIdx++;
        _taskStartTimeUs = nowUs;
        _taskFrameOffsetUs = 0;
        continue;
      }

      int64_t sampleOffsetUs = _taskFrameOffsetUs;
      while (sampleOffsetUs <= elapsedUs && sampleOffsetUs <= task.durationUs) {
        float t = (float)sampleOffsetUs / (float)task.durationUs;
        if (task.mode == TaskMode::EASE)
          t = easeInOut(t);

        applyRampAt(t);
        applyCurrentState();

        if (sampleOffsetUs >= task.durationUs)
          break;
        sampleOffsetUs += _taskStepUs;
        _taskFrameOffsetUs = sampleOffsetUs;
      }

      if (elapsedUs < task.durationUs)
        return;

      applyRampAt(1.0f);
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
