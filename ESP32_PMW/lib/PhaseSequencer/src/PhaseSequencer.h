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

// Add to this for a sub-specification for TaskType:
enum class TaskMode {
  LINEAR,       // constant rate, t
  EASE,      // symmetric S-curve, t^k/(t^k+(1-t)^k); shape k>=1 sharpens
  EXPONENTIAL   // (e^(k*t)-1)/(e^k-1); shape k>0 ease-in, k<0 ease-out
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
  float shape; // curve parameter for EASE/EXPONENTIAL; NAN = per-mode default
};

// Stores the explicit state of all channels at a given microsecond in time
struct TrajectoryPoint {
  int64_t timeUs;
  float freq[4];
  float duty[4];
  float phase[4];
  float carrierDuties[4];
};

// Shared TRAJECTORY_POINT builder (CSV/JSON import).
SequenceTask makeTrajectoryTask(float freq, const float *duty,
                                const float *phase, const float *carrier,
                                int numChannels = 4, int64_t durationUs = 0);

class PhaseSequencer {
public:
  PhaseSequencer(PwmController *phaseCtrl);

  // Queue Builders
  /** @brief Reserve queue capacity for `size` tasks. */
  void reserve(size_t size);
  /** @brief Push a hand-built task (e.g. from makeTrajectoryTask()). */
  void addSequenceTask(SequenceTask task);

  /** @brief Insert a pause of `durationMs` in the sequence. */
  void addWaitTask(uint32_t durationMs);

  /**
   * @brief Ramp one quantity on all channels, start -> end over durationMs
   *        (0 = instant set to end).
   * @param start/end  Hz for PWM_FREQ, else % or degrees.
   * @param type       PWM_FREQ (default), PWM_DUTY, CARRIER_DUTY, PWM_PHASE.
   *                   Duty/carrier clamped 0-100%; PWM_FREQ uses channel 0 only.
   * @param ramp_mode  LINEAR (default), EASE, or EXPONENTIAL.
   * @param shape      Curve parameter (NAN = per-mode default). EASE: S-curve
   *                   sharpness k>=1 (1=linear, 2=default). EXPONENTIAL: exponent
   *                   multiplier k (>0 ease-in, <0 ease-out, 2=default).
   */
  void addRampTask(float start, float end, uint32_t durationMs,
                   TaskType type = TaskType::PWM_FREQ,
                   TaskMode ramp_mode = TaskMode::LINEAR, float shape = NAN);

  /**
   * @brief Per-channel ramp: starts[i] -> ends[i] over durationMs (0 = instant).
   *        NAN in starts[i] leaves that channel unchanged.
   * @param numChannels Entries supplied in starts/ends; the rest are skipped.
   */
  void addRampTask(const float *starts, const float *ends, int numChannels,
                   uint32_t durationMs, TaskType type = TaskType::PWM_FREQ,
                   TaskMode ramp_mode = TaskMode::LINEAR, float shape = NAN);

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
  
  /** @brief Queue index currently running (== queue size once isDone()). Lets
   *  callers track per-step data in parallel with the queue. */
  size_t currentIndex() const { return _currentFrameIdx; }

  /** @brief Carrier duty (%) the schedule last COMMANDED for channel `i`, or NAN
   *  if none yet. This is the sequencer's intent, independent of what wrote the
   *  actual carrier register: a current-balance overlay reads it as the
   *  per-channel ceiling to regulate beneath. Held across WAIT steps. */
  float getCommandedCarrier(int i) const {
    return (i >= 0 && i < 4) ? _currentCarrierDutyCycles[i] : NAN;
  }

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

  // Map linear progress t in [0,1] through the ramp's curve.
  float applyCurve(TaskMode mode, float t, float shape);
};
