#pragma once
#include "../../PhaseController/src/PhaseController.h"
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

// Add to this for a sub-specification for TaskType:
enum class TaskMode {
  LINEAR,
  EASE
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
};

// Stores the explicit state of all channels at a given microsecond in time
struct TrajectoryPoint {
  int64_t timeUs;
  float freq[4];
  float duty[4];
  float phase[4];
  float carrierDuties[4];
};

// Shared TRAJECTORY_POINT builder: used by CSV/JSON import (see
// CsvPhaseSequencer.cpp / JsonPhaseSequencer.cpp).
SequenceTask makeTrajectoryTask(float freq, const float *duty,
                                const float *phase, const float *carrier,
                                int numChannels = 4, int64_t durationUs = 0);

class PhaseSequencer {
public:
  PhaseSequencer(PhaseController *phaseCtrl);

  // Queue Builders
  /** @brief Reserve queue capacity for `size` tasks. */
  void reserve(size_t size);
  /**
   * @brief Push a task built by hand, e.g. a TRAJECTORY_POINT built via
   * makeTrajectoryTask().
   */
  void addSequenceTask(SequenceTask task);

  /** @brief Insert a pause of `durationMs` in the sequence. */
  void addWaitTask(uint32_t durationMs);

  /**
   * @brief Ramp one quantity, all channels identically, start -> end over
   * durationMs (0 = instant set to end).
   *
   * @param start Hz for PWM_FREQ, else % or degrees.
   * @param end Same units as start.
   * @param type PWM_FREQ (default), PWM_DUTY, CARRIER_DUTY or PWM_PHASE.
   *   Duty/carrier are clamped to 0-100%; PWM_FREQ ignores all but channel 0.
   * @param ramp_mode LINEAR (default) or EASE.
   */
  void addRampTask(float start, float end, uint32_t durationMs,
                   TaskType type = TaskType::PWM_FREQ,
                   TaskMode ramp_mode = TaskMode::LINEAR);

  /**
   * @brief Per-channel ramp: starts[i] -> ends[i] over durationMs (0 =
   * instant set to end). NAN in starts[i] leaves that channel unchanged.
   * @param numChannels Entries supplied in starts/ends; the rest are skipped.
   */
  void addRampTask(const float *starts, const float *ends, int numChannels,
                   uint32_t durationMs, TaskType type = TaskType::PWM_FREQ,
                   TaskMode ramp_mode = TaskMode::LINEAR);

  // Compiler
  /**
   * @brief Compile the queue into a trajectory; call before start().
   * @param resolutionMs Trajectory timestep in ms.
   */
  void compile(uint32_t resolutionMs, float initialFreq,
               const float *initialDuty, const float *initialPhase);

  // Control
  void start();
  /** @brief Advance the running sequence. Call every loop() iteration. */
  void run();
  bool isDone() const;
  /** @brief Index of the queue entry currently running (== queue size once
   *  isDone()). Used by callers (e.g. JsonPhaseSequencer::labelForStep) that
   *  track auxiliary per-step data in parallel with the queue. */
  size_t currentIndex() const { return _currentFrameIdx; }

private:
  PhaseController *_phaseCtrl;
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

  float easeInOut(float t);
};
