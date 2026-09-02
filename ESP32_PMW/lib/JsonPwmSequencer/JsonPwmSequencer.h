#pragma once
#include "PwmSequencer.h"
#include <Arduino.h>
#include <vector>

// Forward declaration for ArduinoJson
class JsonVariant;

class JsonPwmSequencer : public PwmSequencer {
public:
  JsonPwmSequencer(PwmController *phaseCtrl);

  /**
   * @brief Load and compile a JSON schedule from SPIFFS. Full schema: README.md.
   *        Object {resolution_ms, initial_freq, initial_duty, direction,
   *        schedule:[...]}; a bare array is the schedule with defaults
   *        (resolution_ms 25, initial_freq 0 = DC, initial_duty {50,50,50,50},
   *        direction CCW).
   * @return False if the file can't be opened or parsed.
   */
  bool loadFromJsonFile(const char *filename);

  /**
   * @brief The `label` in force for the queue step now running, or "" if the
   *        schedule set none. Pushed one per queue entry by the loader, so it
   *        indexes with `currentIndex()`.
   *
   * A schedule-driven experiment sends nothing over serial, so the label printed
   * on change is the ONLY marker the host can align a video against -- see
   * `controller/control/tilt_sweep.py` and `sync.py`.
   */
  const String &stepLabel() const {
    static const String kNone;
    size_t i = currentIndex();
    return i < _stepLabels.size() ? _stepLabels[i] : kNone;
  }

private:
  std::vector<String> _stepLabels;
};
