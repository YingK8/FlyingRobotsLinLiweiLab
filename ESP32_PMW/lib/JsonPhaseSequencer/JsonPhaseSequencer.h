#pragma once
#include "PhaseSequencer.h"
#include <Arduino.h>
#include <vector>

// Forward declaration for ArduinoJson
class JsonVariant;

class JsonPhaseSequencer : public PhaseSequencer {
public:
  JsonPhaseSequencer(PwmController *phaseCtrl);

  /**
   * @brief Load and compile a JSON schedule from SPIFFS. Full schema: README.md.
   *        Object {resolution_ms, initial_freq, initial_duty, direction,
   *        schedule:[...]}; a bare array is the schedule with defaults
   *        (resolution_ms 25, initial_freq 0 = DC, initial_duty {50,50,50,50},
   *        direction CCW).
   * @return False if the file can't be opened or parsed.
   */
  bool loadFromJsonFile(const char *filename);

  /** @brief Telemetry label for queue step `i`, or "" if none / out of range.
   *  Callers poll labelForStep(currentIndex()) each loop() and print on change. */
  const char *labelForStep(size_t i) const;

private:
  std::vector<String> _stepLabels;
};
