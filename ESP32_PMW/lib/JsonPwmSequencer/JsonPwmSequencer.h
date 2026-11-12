#pragma once
#include "PwmSequencer.h"
#include <Arduino.h>
#include <vector>

class JsonVariant;

class JsonPwmSequencer : public PwmSequencer {
public:
  JsonPwmSequencer(PwmController *phaseCtrl);

  // Absent keys fall back to project defaults; a bare array is the schedule.
  bool loadFromJsonFile(const char *filename);

  // The schedule sends nothing over serial; this label is the only host marker.
  const String &stepLabel() const {
    static const String kNone;
    size_t i = currentIndex();
    return i < _stepLabels.size() ? _stepLabels[i] : kNone;
  }

private:
  std::vector<String> _stepLabels;
};
