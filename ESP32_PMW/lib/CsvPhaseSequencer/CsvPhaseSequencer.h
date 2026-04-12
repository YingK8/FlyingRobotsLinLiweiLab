
#pragma once
#include "PhaseController.h"
#include "PhaseSequencer.h"
#include <Arduino.h>
#include <FS.h>
#include <SPIFFS.h>
#include <vector>

const uint32_t STEP_SIZE_MS = 25; // Default time step for trajectory points

class CsvPhaseSequencer : public PhaseSequencer {
public:
  CsvPhaseSequencer(PhaseController *phaseCtrl);

  /**
   * @brief Load a schedule from a CSV file and compile the sequence.
   * @param filename Path to the CSV file.
   * @param resolutionMs Time resolution in milliseconds.
   * @return True if loaded and compiled successfully, false otherwise.
   */
  bool loadFromCSVFile(const char *filename,
                       uint32_t resolutionMs = STEP_SIZE_MS);

  // Existing JSON loader
  bool loadFromJsonFile(const char *filename, uint32_t resolutionMs = 25,
                        float initialFreq = 300.0f,
                        const float *initialDuty = nullptr,
                        const float *initialPhase = nullptr);
};
