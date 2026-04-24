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
  TASK_RAMP_LINEAR,
  TASK_RAMP_EASE,
  TASK_RAMP_PHASE,      // NEW: Phase interpolation
  TASK_TRAJECTORY_POINT // NEW: Directly set a trajectory point (for CSV/JSON
                        // import)
};

struct SequenceTask {
  TaskType type;
  int64_t durationUs;
  float startFreq;
  float endFreq;
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
   * @brief Add a linear frequency ramp task.
   * @param startHz Start frequency in Hz.
   * @param endHz End frequency in Hz.
   * @param durationMs Duration in milliseconds.
   */
  void addLinearRampTask(float startHz, float endHz, uint32_t durationMs);
  /**
   * @brief Add an ease (cubic) frequency ramp task.
   * @param startHz Start frequency in Hz.
   * @param endHz End frequency in Hz.
   * @param durationMs Duration in milliseconds.
   */
  void addEaseRampTask(float startHz, float endHz, uint32_t durationMs);
  /**
   * @brief Add a phase ramp task for all channels.
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
  std::vector<TrajectoryPoint> _trajectory;

  size_t _currentFrameIdx;
  int64_t _taskStartTimeUs;

  float easeInOut(float t);
};