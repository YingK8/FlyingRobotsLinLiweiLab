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
   * @brief Load a JSON schedule (SPIFFS) and compile it. The file is an object
   *        carrying the initial state plus the schedule:
   *        {
   *          "resolution_ms": 25, "initial_freq": 190.0,
   *          "initial_duty": [50,50,50,50], "direction": "CCW",
   *          "schedule": [ {method, channel, mask, value/from/to, duration_ms}, ... ]
   *        }
   *        All config keys are optional (defaults: resolution_ms 25,
   *        initial_freq 0 (DC/stationary), initial_duty {50,50,50,50},
   *        direction CCW). A bare
   *        top-level array is still accepted as the schedule with those
   *        defaults. No loop/repeat (unrolled by the generator). Full schema:
   *        README.md.
   * @return False if the file can't be opened or parsed.
   */
  bool loadFromJsonFile(const char *filename);

  /** @brief Telemetry label for queue step `i`, or "" if none / out of range.
   *  Callers poll labelForStep(currentIndex()) each loop() and print on change. */
  const char *labelForStep(size_t i) const;

private:
  std::vector<String> _stepLabels;
};
