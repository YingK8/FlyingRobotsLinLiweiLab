#include "JsonPWMSequencer.h"
#include <ArduinoJson.h>
#include <FS.h>
#include <SPIFFS.h>
#include <math.h>

namespace {
// Project-wide rotation conventions (see lib/JsonPWMSequencer/README.md):
// coil order is A,B,C,D.
const float PHASES_CW[4] = {270.0f, 90.0f, 180.0f, 0.0f};
const float PHASES_CCW[4] = {90.0f, 270.0f, 180.0f, 0.0f};
} // namespace

JsonPWMSequencer::JsonPWMSequencer(PWMController *phaseCtrl)
    : PWMSequencer(phaseCtrl) {}

const char *JsonPWMSequencer::labelForStep(size_t i) const {
  if (i >= _stepLabels.size())
    return "";
  return _stepLabels[i].c_str();
}

bool JsonPWMSequencer::loadFromJsonFile(const char *filename,
                                          uint32_t resolutionMs,
                                          float initialFreq,
                                          const float *initialDuty,
                                          const float *initialPhase) {
  File file = SPIFFS.open(filename, "r");
  if (!file)
    return false;
  size_t size = file.size();
  std::unique_ptr<char[]> buf(new char[size + 1]);
  file.readBytes(buf.get(), size);
  buf[size] = '\0';
  file.close();

  // Elastic document (ArduinoJson v7): grows its internal pool as needed, so
  // there's no fixed byte-capacity to guess for a large generated experiment.
  JsonDocument doc;
  auto err = deserializeJson(doc, buf.get());
  if (err) {
    Serial.printf("[JsonPWMSequencer] parse failed: %s (file=%u bytes, "
                  "free heap=%u bytes)\n",
                  err.c_str(), (unsigned)size, (unsigned)ESP.getFreeHeap());
    return false;
  }

  JsonArray arr = doc.as<JsonArray>();
  reserve(arr.size());
  _stepLabels.reserve(arr.size());

  // Running full state: TRAJECTORY_POINT tasks need every channel, so each
  // per-channel command updates one entry here and pushes the whole snapshot.
  float defaultDuty[4] = {50, 50, 50, 50};
  float defaultPhase[4] = {0, 90, 180, 270};
  if (!initialDuty)
    initialDuty = defaultDuty;
  if (!initialPhase)
    initialPhase = defaultPhase;
  float curFreq = initialFreq;
  float curDuty[4];
  float curPhase[4];
  float curCarrier[4]; // NAN = untouched; applyCurrentState() skips those channels
  for (int i = 0; i < 4; i++) {
    curDuty[i] = initialDuty[i];
    curPhase[i] = initialPhase[i];
    curCarrier[i] = NAN;
  }

  // Label active for whatever step gets pushed next; "label" entries update
  // this without pushing a queue entry of their own.
  String currentLabel;

  std::vector<String> unknownMethods;
  for (JsonObject obj : arr) {
    const char *methodCStr = obj["method"] | "";
    String method(methodCStr);
    int channel = obj["channel"] | -1;
    int mask = obj["mask"] | 0;
    float value = obj["value"] | 0.0f;
    float from = obj["from"] | 0.0f;
    float to = obj["to"] | 0.0f;
    uint32_t durationMs = obj["duration_ms"] | 0;
    bool called = false;
    bool pushedTask = false;
    auto hasValidChannel = [&]() { return channel >= 0 && channel < 4; };

    if (method == "addDutyCycleTask" && hasValidChannel()) {
      curDuty[channel] = constrain(value, 0.0f, 100.0f);
      addSequenceTask(makeTrajectoryTask(curFreq, curDuty, curPhase, curCarrier));
      called = true;
      pushedTask = true;
    } else if (method == "addPhaseTask" && hasValidChannel()) {
      curPhase[channel] = value;
      addSequenceTask(makeTrajectoryTask(curFreq, curDuty, curPhase, curCarrier));
      called = true;
      pushedTask = true;
    } else if (method == "addWaitTask") {
      addWaitTask(durationMs);
      called = true;
      pushedTask = true;
    } else if (method == "addLinearRampTask") {
      addRampTask(from, to, durationMs, TaskType::PWM_FREQ, TaskMode::LINEAR);
      curFreq = to;
      called = true;
      pushedTask = true;
    } else if (method == "addEaseRampTask") {
      addRampTask(from, to, durationMs, TaskType::PWM_FREQ, TaskMode::EASE);
      curFreq = to;
      called = true;
      pushedTask = true;
    } else if (method == "addCarrierRampTask") {
      addRampTask(from, to, durationMs, TaskType::CARRIER_DUTY, TaskMode::LINEAR);
      for (int i = 0; i < 4; i++)
        curCarrier[i] = to;
      called = true;
      pushedTask = true;
    } else if (method == "addCarrierEaseRampTask") {
      addRampTask(from, to, durationMs, TaskType::CARRIER_DUTY, TaskMode::EASE);
      for (int i = 0; i < 4; i++)
        curCarrier[i] = to;
      called = true;
      pushedTask = true;
    } else if (method == "addPhaseRampTask" && hasValidChannel()) {
      // Ramp only the named channel; NAN leaves the others alone.
      float starts[4] = {NAN, NAN, NAN, NAN};
      float ends[4] = {NAN, NAN, NAN, NAN};
      starts[channel] = from;
      ends[channel] = to;
      addRampTask(starts, ends, 4, durationMs, TaskType::PWM_PHASE,
                  TaskMode::EASE);
      curPhase[channel] = to;
      called = true;
      pushedTask = true;
    } else if (method == "addCarrierDutyCycleTask" && hasValidChannel()) {
      curCarrier[channel] = constrain(value, 0.0f, 100.0f);
      addSequenceTask(makeTrajectoryTask(curFreq, curDuty, curPhase, curCarrier));
      called = true;
      pushedTask = true;
    } else if (method == "setDirection") {
      // value != 0 => CCW, else CW (see PHASES_CW/PHASES_CCW above).
      const float *phases = (value != 0.0f) ? PHASES_CCW : PHASES_CW;
      for (int i = 0; i < 4; i++)
        curPhase[i] = phases[i];
      addSequenceTask(makeTrajectoryTask(curFreq, curDuty, curPhase, curCarrier));
      called = true;
      pushedTask = true;
    } else if (method == "activateChannels") {
      // "mask" bit i set => channel i carrier duty = value (clamped); else 0.
      float onDuty = constrain(value, 0.0f, 100.0f);
      for (int i = 0; i < 4; i++)
        curCarrier[i] = ((mask >> i) & 1) ? onDuty : 0.0f;
      addSequenceTask(makeTrajectoryTask(curFreq, curDuty, curPhase, curCarrier));
      called = true;
      pushedTask = true;
    } else if (method == "label") {
      const char *labelCStr = obj["value"] | "";
      currentLabel = String(labelCStr);
      called = true; // recognized, but pushes nothing -- doesn't advance the queue
    }

    if (pushedTask) {
      _stepLabels.push_back(currentLabel);
    }
    if (!called) {
      unknownMethods.push_back(method);
    }
  }

  if (!unknownMethods.empty()) {
    Serial.println("[JsonPWMSequencer] Unknown methods found in schedule:");
    for (const auto &m : unknownMethods) {
      Serial.println(m);
    }
  }

  compile(resolutionMs, initialFreq, initialDuty, initialPhase);
  return true;
}
