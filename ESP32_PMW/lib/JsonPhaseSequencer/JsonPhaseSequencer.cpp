#include "JsonPhaseSequencer.h"
#include <ArduinoJson.h>
#include <FS.h>
#include <SPIFFS.h>

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

  // Clear any existing queue
  // (Assumes PhaseSequencer exposes _queue or provides a clear method)
  // If not, just create a new instance or add a clearQueue() method.
  // For now, assume we can just append for demonstration.

  std::vector<String> unknownMethods;
  for (JsonObject obj : doc.as<JsonArray>()) {
    const char *methodCStr = obj["method"] | "";
    String method(methodCStr);
    int channel = obj["channel"] | -1;
    float value = obj["value"] | 0.0f;
    float from = obj["from"] | 0.0f;
    float to = obj["to"] | 0.0f;
    uint32_t durationMs = obj["duration_ms"] | 0;
    float arr4[4] = {0, 0, 0, 0};
    bool called = false;
    auto hasValidChannel = [&]() { return channel >= 0 && channel < 4; };

    if (method == "addDutyCycleTask" && hasValidChannel()) {
      arr4[channel] = value;
      addDutyCycleTask(arr4, 4);
      called = true;
    } else if (method == "addPhaseTask" && hasValidChannel()) {
      arr4[channel] = value;
      addPhaseTask(arr4, 4);
      called = true;
    } else if (method == "addWaitTask") {
      addWaitTask(durationMs);
      called = true;
    } else if (method == "addLinearRampTask") {
      addLinearRampTask(from, to, durationMs);
      called = true;
    } else if (method == "addEaseRampTask") {
      addEaseRampTask(from, to, durationMs);
      called = true;
    } else if (method == "addPhaseRampTask" && hasValidChannel()) {
      float startPhases[4] = {0, 0, 0, 0};
      float endPhases[4] = {0, 0, 0, 0};
      startPhases[channel] = from;
      endPhases[channel] = to;
      addPhaseRampTask(startPhases, endPhases, durationMs);
      called = true;
    } else if (method == "addCarrierDutyCycleTask" && hasValidChannel()) {
      arr4[channel] = value;
      addCarrierDutyCycleTask(arr4, 4);
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

  // Compile the trajectory
  float defaultDuty[4] = {50, 50, 50, 50};
  float defaultPhase[4] = {0, 90, 180, 270};
  if (!initialDuty)
    initialDuty = defaultDuty;
  if (!initialPhase)
    initialPhase = defaultPhase;
  compile(resolutionMs, initialFreq, initialDuty, initialPhase);
  return true;
}
