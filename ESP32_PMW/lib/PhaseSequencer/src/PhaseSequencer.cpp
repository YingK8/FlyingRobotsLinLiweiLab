#include "PhaseSequencer.h"

PhaseSequencer::PhaseSequencer(PhaseController *phaseCtrl) {
  _phaseCtrl = phaseCtrl;
  _currentFrameIdx = 0;
  _taskStartTimeUs = 0;
}

void PhaseSequencer::reserve(size_t size) { _queue.reserve(size); }

// Updated builders to zero-initialize the new struct fields automatically
void PhaseSequencer::addDutyCycleTask(const float *dutyCycles,
                                      int numChannels) {
  SequenceTask task = {}; // Zero-initializes all fields
  task.type = TASK_SET_DUTY_CYCLES;
  for (int i = 0; i < numChannels; i++) {
    task.dutyCycles[i] = constrain(dutyCycles[i], 0.0, 100.0);
  }
  _queue.push_back(task);
}

// Add a carrier duty cycle set task for all channels.
void PhaseSequencer::addCarrierDutyCycleTask(const float *carrierDutyCycles,
                                             int numChannels) {
  SequenceTask task = {}; // Zero-initializes all fields
  task.type = TASK_SET_DUTY_CYCLES;
  for (int i = 0; i < numChannels; i++) {
    task.carrierDuties[i] = constrain(carrierDutyCycles[i], 0.0, 100.0);
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

float PhaseSequencer::easeInOut(float t) {
  if (t < 0.0f)
    t = 0.0f;
  if (t > 1.0f)
    t = 1.0f;
  return t * t * (3.0f - 2.0f * t);
}

void PhaseSequencer::compile(uint32_t resolutionMs, float initialFreq,
                             const float *initialDuty,
                             const float *initialPhase) {
  _trajectory.clear();

  int64_t currentTimeUs = 0;
  float curFreq[4] = {initialFreq, initialFreq, initialFreq, initialFreq};
  float curDuty[4] = {initialDuty[0], initialDuty[1], initialDuty[2],
                      initialDuty[3]};
  float curPhase[4] = {initialPhase[0], initialPhase[1], initialPhase[2],
                       initialPhase[3]};

  for (const auto &task : _queue) {
    if (task.type == TASK_SET_DUTY_CYCLES) {
      for (int i = 0; i < 4; i++) {
        curDuty[i] = task.dutyCycles[i];
        // Carrier duty
        pt.carrierDuties[i] = task.carrierDuties[i];
      }
      TrajectoryPoint pt;
      pt.timeUs = currentTimeUs;
      for (int i = 0; i < 4; i++) {
        pt.freq[i] = curFreq[i];
        pt.duty[i] = curDuty[i];
        pt.phase[i] = curPhase[i];
        pt.carrierDuties[i] = task.carrierDuties[i];
      }
      _trajectory.push_back(pt);

    } else if (task.type == TASK_SET_PHASES) {
      for (int i = 0; i < 4; i++)
        curPhase[i] = task.endPhases[i];

      TrajectoryPoint pt;
      pt.timeUs = currentTimeUs;
      for (int i = 0; i < 4; i++) {
        pt.freq[i] = curFreq[i];
        pt.duty[i] = curDuty[i];
        pt.phase[i] = curPhase[i];
      }
      _trajectory.push_back(pt);

    } else if (task.type == TASK_WAIT) {
      currentTimeUs += task.durationUs;

      TrajectoryPoint pt;
      pt.timeUs = currentTimeUs;
      for (int i = 0; i < 4; i++) {
        pt.freq[i] = curFreq[i];
        pt.duty[i] = curDuty[i];
        pt.phase[i] = curPhase[i];
      }
      _trajectory.push_back(pt);

    } else if (task.type == TASK_RAMP_LINEAR || task.type == TASK_RAMP_EASE) {
      int64_t stepUs = (int64_t)resolutionMs * 1000LL;
      int64_t elapsed = 0;

      while (elapsed <= task.durationUs) {
        float t = (float)elapsed / task.durationUs;
        if (task.type == TASK_RAMP_EASE)
          t = easeInOut(t);

        for (int i = 0; i < 4; i++) {
          curFreq[i] = task.startFreq + t * (task.endFreq - task.startFreq);
        }

        TrajectoryPoint pt;
        pt.timeUs = currentTimeUs + elapsed;
        for (int i = 0; i < 4; i++) {
          pt.freq[i] = curFreq[i];
          pt.duty[i] = curDuty[i];
          pt.phase[i] = curPhase[i];
        }
        _trajectory.push_back(pt);

        elapsed += stepUs;
      }

      for (int i = 0; i < 4; i++)
        curFreq[i] = task.endFreq;
      currentTimeUs += task.durationUs;

      TrajectoryPoint pt;
      pt.timeUs = currentTimeUs;
      for (int i = 0; i < 4; i++) {
        pt.freq[i] = curFreq[i];
        pt.duty[i] = curDuty[i];
        pt.phase[i] = curPhase[i];
      }
      _trajectory.push_back(pt);

    } else if (task.type == TASK_RAMP_PHASE) {
      int64_t stepUs = (int64_t)resolutionMs * 1000LL;
      int64_t elapsed = 0;

      while (elapsed <= task.durationUs) {
        float t = (float)elapsed / task.durationUs;

        // We ease-in/ease-out the phase angle shift to prevent magnetic "jerks"
        // when the field starts morphing from an ellipse to a circle.
        t = easeInOut(t);

        for (int i = 0; i < 4; i++) {
          // Linear interpolation handles the 0->90 and 180->270 shifts
          // perfectly.
          curPhase[i] = task.startPhases[i] +
                        t * (task.endPhases[i] - task.startPhases[i]);
        }

        TrajectoryPoint pt;
        pt.timeUs = currentTimeUs + elapsed;
        for (int i = 0; i < 4; i++) {
          pt.freq[i] = curFreq[i];
          pt.duty[i] = curDuty[i];
          pt.phase[i] = curPhase[i];
        }
        _trajectory.push_back(pt);

        elapsed += stepUs;
      }

      for (int i = 0; i < 4; i++)
        curPhase[i] = task.endPhases[i];
      currentTimeUs += task.durationUs;

      TrajectoryPoint pt;
      pt.timeUs = currentTimeUs;
      for (int i = 0; i < 4; i++) {
        pt.freq[i] = curFreq[i];
        pt.duty[i] = curDuty[i];
        pt.phase[i] = curPhase[i];
      }
      _trajectory.push_back(pt);
    }
  }
}

void PhaseSequencer::start() {
  if (_trajectory.empty())
    return;
  _currentFrameIdx = 0;
  _taskStartTimeUs = esp_timer_get_time();
}

bool PhaseSequencer::isDone() const {
  return _currentFrameIdx >= _trajectory.size();
}

// =========================================================
// HIGH-SPEED HOT LOOP: No math, just pointer lookups
// =========================================================
void PhaseSequencer::run() {
  if (_trajectory.empty())
    return;

  int64_t nowUs = esp_timer_get_time() - _taskStartTimeUs;

  while (_currentFrameIdx < _trajectory.size() &&
         nowUs >= _trajectory[_currentFrameIdx].timeUs) {

    TrajectoryPoint &pt = _trajectory[_currentFrameIdx];

    _phaseCtrl->setGlobalFrequency(pt.freq[0]);

    for (int i = 0; i < 4; i++) {
      _phaseCtrl->setDutyCycle(i, pt.duty[i]);
      _phaseCtrl->setPhase(i, pt.phase[i]);
    }

    // Set carrier duty cycle if present
    for (int i = 0; i < 4; i++) {
      if (pt.carrierDuties[i] > 0.0f || pt.carrierDuties[i] == 0.0f) // allow 0%
        _phaseCtrl->setCarrierDutyCycle(i, pt.carrierDuties[i]);
    }

    _currentFrameIdx++;
  }
}