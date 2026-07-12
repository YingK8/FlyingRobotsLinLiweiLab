#pragma once
#include "PWMSequencer.h"
#include <Arduino.h>
#include <vector>

// Forward declaration for ArduinoJson
class JsonVariant;

class JsonPWMSequencer : public PWMSequencer {
public:
  JsonPWMSequencer(PWMController *phaseCtrl);

  // Load a schedule from a JSON file (SPIFFS) and compile it. File is a flat
  // JSON array of {method, channel, mask, value/from/to, duration_ms}
  // entries, no loop/repeat primitive (repeats are unrolled by whoever
  // generates the file). Full method list and schema: README.md.
  // initialDuty: float[4] duty (0-100%), defaults to {50,50,50,50}.
  // initialPhase: float[4] phase (degrees), defaults to {0,90,180,270}.
  // Returns false if the file can't be opened or fails to parse.
  bool loadFromJsonFile(const char *filename, uint32_t resolutionMs = 25,
                        float initialFreq = 300.0f,
                        const float *initialDuty = nullptr,
                        const float *initialPhase = nullptr);

  // Telemetry label active during queue step `i` (see
  // PWMSequencer::currentIndex()), or "" if none was set / `i` is out of
  // range. Callers typically poll labelForStep(currentIndex()) once per
  // loop() and print it on change.
  const char *labelForStep(size_t i) const;

private:
  std::vector<String> _stepLabels;
};
