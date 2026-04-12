#pragma once
#include "CsvPhaseSequencer.h"
#include <Arduino.h>
#include <vector>

const uint32_t STEP_SIZE_MS = 25; // Default time step for trajectory points (can be overridden in loadFromCSVFile)

class CsvPhaseSequencer : public PhaseSequencer {
public:
  /**
   * @brief Construct a CsvPhaseSequencer for JSON-based scheduling.
   * @param phaseCtrl Pointer to an initialized PhaseController.
   */
  CsvPhaseSequencer(PhaseController *phaseCtrl);
  /**
   * @brief Load a schedule from a CSV file and compile the sequence.
   * @param filename Path to the CSV file.
   * @param resolutionMs Time resolution in milliseconds.
   * @param initialFreq Initial frequency in Hz.
   * @param initialDuty Array of initial duty cycles.
   * @param initialPhase Array of initial phases.
   * @return True if loaded and compiled successfully, false otherwise.
   */
  // Load schedule from JSON file (on SD or SPIFFS)
  bool loadFromJsonFile(const char *filename, uint32_t resolutionMs = 25,
                        float initialFreq = 300.0f,
                        const float *initialDuty = nullptr,
                        const float *initialPhase = nullptr);

  /**
   * @brief Schedule a carrier duty cycle change using a JSON method entry.
   * Example: { "method": "addCarrierDutyCycleTask", "channel": 0, "value": 75.0
   * }
   */
};
