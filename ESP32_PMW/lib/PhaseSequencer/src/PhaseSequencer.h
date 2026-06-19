#pragma once
#include "../../PhaseController/src/PhaseController.h"
#include "esp_timer.h"
#include <Arduino.h>
#include <vector>

// Define the types of commands our queue can execute
enum TaskType {
  TASK_SET_DUTY_CYCLES,
  TASK_SET_CARRIER_DUTY_CYCLES,
  TASK_SET_PHASES, // NEW: Instant phase snap
  TASK_WAIT,
  TASK_RAMP_LINEAR,     // Ramp with linear interpolation
  TASK_RAMP_EASE,       // Ramp with cubic ease-in/out interpolation
  TASK_TRAJECTORY_POINT // NEW: Directly set a trajectory point (for CSV/JSON
                        // import)
};

// For ramp tasks, which quantity is being interpolated. (`class` is a reserved
// keyword in C++, so the SequenceTask field is named `cls`.)
enum RampTarget {
  RAMP_PWM,     // global PWM frequency (Hz)
  RAMP_CARRIER, // carrier duty cycle (%) on all channels
  RAMP_PHASE,   // per-channel phase (degrees)
};

struct SequenceTask {
  TaskType type;
  RampTarget cls; // for TASK_RAMP_* tasks: what is being ramped
  int64_t durationUs;
  float startFreq;
  float endFreq;
  float startCarriers[4];
  float endCarriers[4];
  float dutyCycles[4];
  float carrierDuties[4];
  float startPhases[4];
  float endPhases[4];
};

// Stores the explicit state of all channels at a given microsecond in time
struct TrajectoryPoint {
  int64_t timeUs;
  float freq[4];
  float duty[4];
  float phase[4];
  float carrierDuties[4];
};

class PhaseSequencer {
public:
  /**
   * @brief Construct a PhaseSequencer for high-level PWM sequencing.
   * @param phaseCtrl Pointer to an initialized PhaseController.
   */
  PhaseSequencer(PhaseController *phaseCtrl);

  // Queue Builders
  /**
   * @brief Reserve space for a number of tasks in the queue.
   * @param size Number of tasks to reserve.
   */
  void reserve(size_t size);
  /**
   * @brief Add a generic sequence task to the queue.
   * @param pt The SequenceTask to add.
   */
  void addSequenceTask(SequenceTask task);
  /**
   * @brief Add a duty cycle set task for all channels.
   * @param dutyCycles Array of duty cycles.
   * @param numChannels Number of channels.
   */
  void addDutyCycleTask(const float *dutyCycles, int numChannels);
  /**
   * @brief Add a carrier duty cycle set task for all channels.
   * @param carrierDutyCycles Array of carrier duty cycles (0-100%).
   * @param numChannels Number of channels.
   */
  void addCarrierDutyCycleTask(const float *carrierDutyCycles, int numChannels);
  /**
   * @brief Add a phase set task for all channels.
   * @param phases Array of phases (degrees).
   * @param numChannels Number of channels.
   */
  void addPhaseTask(const float *phases, int numChannels);
  /**
   * @brief Add a wait task (pause) in the sequence.
   * @param durationMs Duration in milliseconds.
   */
  void addWaitTask(uint32_t durationMs);
  /**
   * @brief Add a PWM frequency ramp task.
   * @param startHz Start frequency in Hz.
   * @param endHz End frequency in Hz.
   * @param durationMs Duration in milliseconds.
   * @param type Ramp interpolation type: TASK_RAMP_LINEAR (default) or
   *             TASK_RAMP_EASE.
   */
  void addPWMRampTask(float startHz, float endHz, uint32_t durationMs,
                   TaskType type = TASK_RAMP_LINEAR);
  /**
   * @brief Add a carrier duty cycle ramp task (applied to all channels).
   * @param startDuty Start carrier duty cycle (0-100%).
   * @param endDuty End carrier duty cycle (0-100%).
   * @param durationMs Duration in milliseconds.
   * @param type Ramp interpolation type: TASK_RAMP_LINEAR (default) or
   *             TASK_RAMP_EASE.
   */
  void addCarrierRampTask(float startDuty, float endDuty, uint32_t durationMs,
                          TaskType type = TASK_RAMP_LINEAR);
  /**
   * @brief Add a phase ramp task for all channels (cubic ease).
   * @param startPhases Array of start phases (degrees).
   * @param endPhases Array of end phases (degrees).
   * @param durationMs Duration in milliseconds.
   */
  void addPhaseRampTask(const float *startPhases, const float *endPhases,
                        uint32_t durationMs);

  // Compiler
  /**
   * @brief Compile the queued tasks into a trajectory.
   * @param resolutionMs Time resolution in milliseconds.
   * @param initialFreq Initial frequency in Hz.
   * @param initialDuty Array of initial duty cycles.
   * @param initialPhase Array of initial phases.
   */
  void compile(uint32_t resolutionMs, float initialFreq,
               const float *initialDuty, const float *initialPhase);

  // Control
  /**
   * @brief Start executing the compiled sequence.
   */
  void start();
  /**
   * @brief Run the sequencer. Call regularly in loop().
   */
  void run();
  /**
   * @brief Check if the sequence is done.
   * @return True if done, false otherwise.
   */
  bool isDone() const;

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