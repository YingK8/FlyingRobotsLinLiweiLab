#pragma once
#include "../../PwmController/src/PwmController.h"
#include "esp_timer.h"
#include <Arduino.h>
#include <vector>

enum class TaskType {
  PWM_DUTY,
  PWM_FREQ,
  PWM_PHASE,
  CARRIER_DUTY,
  WAIT,
  TRAJECTORY_POINT
};

enum class TaskMode {
  POLYNOMIAL,
  EASE,
  EXPONENTIAL
};

struct SequenceTask {
  TaskType type;
  TaskMode mode;
  int64_t durationUs;
  float startFreq;
  float endFreq;
  float startCarriers[4];
  float endCarriers[4];
  float startDuties[4];
  float endDuties[4];
  float startPhases[4];
  float endPhases[4];
  float dutyCycles[4];
  float carrierDuties[4];
  float shape;
};

struct TrajectoryPoint {
  int64_t timeUs;
  float freq[4];
  float duty[4];
  float phase[4];
  float carrierDuties[4];
};

// Sets all four channels at one instant; frequency is global.
SequenceTask makeTrajectoryTask(float freq, const float *duty,
                                const float *phase, const float *carrier,
                                int numChannels = 4, int64_t durationUs = 0);

class PwmSequencer {
public:
  PwmSequencer(PwmController *phaseCtrl);

  void reserve(size_t size);

  // Drops every queued step; the caller must not do this mid-run.
  void clear();
  void addSequenceTask(SequenceTask task);

  void addWaitTask(uint32_t durationMs);

  // Duty and carrier clamp to 0-100 percent; frequency uses channel 0 only.
  void addRampTask(float start, float end, uint32_t durationMs,
                   TaskType type = TaskType::PWM_FREQ,
                   TaskMode ramp_mode = TaskMode::POLYNOMIAL, float shape = NAN);

  // NAN in starts leaves that channel unchanged; 0 ms sets the end at once.
  void addRampTask(const float *starts, const float *ends, int numChannels,
                   uint32_t durationMs, TaskType type = TaskType::PWM_FREQ,
                   TaskMode ramp_mode = TaskMode::POLYNOMIAL, float shape = NAN);

  // Call before start(): sets the initial state and the sample period.
  void compile(uint32_t resolutionMs, float initialFreq,
               const float *initialDuty, const float *initialPhase);

  // Call every loop() iteration, or the trajectory falls behind the clock.
  void start();

  void run();

  bool isDone() const;

  // Equals the step count once hasFinished(), so callers can track per-step data.
  size_t currentIndex() const { return _currentFrameIdx; }


private:
  PwmController *_phaseCtrl;
  std::vector<SequenceTask> _queue;
  float _initialFreqHz;
  float _initialDutyCycles[4];
  float _initialPhaseDegrees[4];
  float _currentFreqHz;
  float _currentDutyCycles[4];
  float _currentPhaseDegrees[4];
  float _currentCarrierDutyCycles[4];
  size_t _currentFrameIdx;
  int64_t _taskStartTimeUs;
  int64_t _taskFrameOffsetUs;
  int64_t _taskStepUs;

  void resetStreamingState();
  void applyCurrentState();

  // Maps linear progress in [0,1] through the step's ramp mode.
  float applyCurve(TaskMode mode, float t, float shape);
};
