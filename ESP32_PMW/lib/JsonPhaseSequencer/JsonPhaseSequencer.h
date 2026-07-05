#pragma once
#include "PhaseSequencer.h"
#include <Arduino.h>
#include <vector>

// Forward declaration for ArduinoJson
class JsonVariant;

class JsonPhaseSequencer : public PhaseSequencer {
public:
  JsonPhaseSequencer(PhaseController *phaseCtrl);

  /**
   * @brief Load a schedule from a JSON file (SPIFFS) and compile it.
   *
   * The file is a JSON array of entries: { "method": ..., "channel": ...,
   * "value"/"from"/"to": ..., "duration_ms": ... }, e.g.
   *   { "method": "addCarrierDutyCycleTask", "channel": 0, "value": 75.0 }
   * `method` is one of addDutyCycleTask / addPhaseTask / addCarrierDutyCycleTask
   * (instant, per-channel set) or addWaitTask / addLinearRampTask /
   * addEaseRampTask / addCarrierRampTask / addCarrierEaseRampTask /
   * addPhaseRampTask (see loadFromJsonFile in the .cpp for the full mapping).
   *
   * @param initialDuty float[4] duty (0-100%); defaults to {50,50,50,50}.
   * @param initialPhase float[4] phase (degrees); defaults to {0,90,180,270}.
   * @return False if the file can't be opened or fails to parse.
   */
  bool loadFromJsonFile(const char *filename, uint32_t resolutionMs = 25,
                        float initialFreq = 300.0f,
                        const float *initialDuty = nullptr,
                        const float *initialPhase = nullptr);
};
