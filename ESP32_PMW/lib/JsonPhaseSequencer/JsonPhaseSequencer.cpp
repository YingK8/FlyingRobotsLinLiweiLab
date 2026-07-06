#include "JsonPhaseSequencer.h"
#include <ArduinoJson.h>
#include <FS.h>
#include <SPIFFS.h>
#include <math.h>

JsonPhaseSequencer::JsonPhaseSequencer(PhaseController *phaseCtrl)
    : PhaseSequencer(phaseCtrl) {}

bool JsonPhaseSequencer::loadFromJsonFile(const char *filename,
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

  DynamicJsonDocument doc(4096);
  auto err = deserializeJson(doc, buf.get());
  if (err)
    return false;

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

  std::vector<String> unknownMethods;
  for (JsonObject obj : doc.as<JsonArray>()) {
    const char *methodCStr = obj["method"] | "";
    String method(methodCStr);
    int channel = obj["channel"] | -1;
    float value = obj["value"] | 0.0f;
    float from = obj["from"] | 0.0f;
    float to = obj["to"] | 0.0f;
    uint32_t durationMs = obj["duration_ms"] | 0;
    bool called = false;
    auto hasValidChannel = [&]() { return channel >= 0 && channel < 4; };

    if (method == "addDutyCycleTask" && hasValidChannel()) {
      curDuty[channel] = constrain(value, 0.0f, 100.0f);
      addSequenceTask(makeTrajectoryTask(curFreq, curDuty, curPhase, curCarrier));
      called = true;
    } else if (method == "addPhaseTask" && hasValidChannel()) {
      curPhase[channel] = value;
      addSequenceTask(makeTrajectoryTask(curFreq, curDuty, curPhase, curCarrier));
      called = true;
    } else if (method == "addWaitTask") {
      addWaitTask(durationMs);
      called = true;
    } else if (method == "addLinearRampTask") {
      addRampTask(from, to, durationMs, TaskType::PWM_FREQ, TaskMode::LINEAR);
      curFreq = to;
      called = true;
    } else if (method == "addEaseRampTask") {
      addRampTask(from, to, durationMs, TaskType::PWM_FREQ, TaskMode::EASE);
      curFreq = to;
      called = true;
    } else if (method == "addCarrierRampTask") {
      addRampTask(from, to, durationMs, TaskType::CARRIER_DUTY, TaskMode::LINEAR);
      for (int i = 0; i < 4; i++)
        curCarrier[i] = to;
      called = true;
    } else if (method == "addCarrierEaseRampTask") {
      addRampTask(from, to, durationMs, TaskType::CARRIER_DUTY, TaskMode::EASE);
      for (int i = 0; i < 4; i++)
        curCarrier[i] = to;
      called = true;
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
    } else if (method == "addCarrierDutyCycleTask" && hasValidChannel()) {
      curCarrier[channel] = constrain(value, 0.0f, 100.0f);
      addSequenceTask(makeTrajectoryTask(curFreq, curDuty, curPhase, curCarrier));
      called = true;
    }
    if (!called) {
      unknownMethods.push_back(method);
    }
  }

  if (!unknownMethods.empty()) {
    Serial.println("[JsonPhaseSequencer] Unknown methods found in schedule:");
    for (const auto &m : unknownMethods) {
      Serial.println(m);
    }
  }

  compile(resolutionMs, initialFreq, initialDuty, initialPhase);
  return true;
}
