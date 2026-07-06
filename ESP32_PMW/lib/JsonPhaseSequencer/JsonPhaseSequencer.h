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
   * @brief Load a schedule from a JSON file (SPIFFS) and compile it. File is
   * a flat JSON array of {method, channel, mask, value/from/to, duration_ms}
   * entries, no loop/repeat primitive (repeats are unrolled by whoever
   * generates the file). Full method list and schema: README.md.
   * @param initialDuty float[4] duty (0-100%); defaults to {50,50,50,50}.
   * @param initialPhase float[4] phase (degrees); defaults to {0,90,180,270}.
   * @return False if the file can't be opened or fails to parse.
   */
  bool loadFromJsonFile(const char *filename, uint32_t resolutionMs = 25,
                        float initialFreq = 300.0f,
                        const float *initialDuty = nullptr,
                        const float *initialPhase = nullptr);

  /** @brief The telemetry label active during queue step `i` (see
   *  PhaseSequencer::currentIndex()), or "" if none was set / `i` is out of
   *  range. Callers typically poll labelForStep(currentIndex()) once per
   *  loop() and print it on change. */
  const char *labelForStep(size_t i) const;

private:
  std::vector<String> _stepLabels;
};
