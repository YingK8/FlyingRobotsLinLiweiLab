#include "JsonPwmSequencer.h"
#include <ArduinoJson.h>
#include <FS.h>
#include <SPIFFS.h>
#include <math.h>
#include <strings.h> // strcasecmp

namespace {
// Operator-confirmed A,B,C,D order; A/C and B/D oppose; reversal swaps B/D.
const float PHASES_CW[4] = {0.0f, 90.0f, 180.0f, 270.0f};
const float PHASES_CCW[4] = {0.0f, 270.0f, 180.0f, 90.0f};
} // namespace

JsonPwmSequencer::JsonPwmSequencer(PwmController *phaseCtrl)
    : PwmSequencer(phaseCtrl) {}

bool JsonPwmSequencer::loadFromJsonFile(const char *filename) {
  File file = SPIFFS.open(filename, "r");
  if (!file) {
    Serial.printf("[JsonPwmSequencer] cannot open %s -- is SPIFFS mounted "
                  "and the filesystem image uploaded (pio run -t uploadfs)?\n",
                  filename);
    return false;
  }
  size_t size = file.size();
  // Elastic document: no fixed pool size to guess for a large schedule.
  std::unique_ptr<char[]> buf(new char[size + 1]);
  file.readBytes(buf.get(), size);
  buf[size] = '\0';
  file.close();

  JsonDocument doc;
  auto err = deserializeJson(doc, buf.get());
  if (err) {
    Serial.printf("[JsonPwmSequencer] parse failed: %s (file=%u bytes, "
                  "free heap=%u bytes)\n",
                  err.c_str(), (unsigned)size,
                  (unsigned)ESP.getFreeHeap());
    return false;
  }

  uint32_t resolutionMs = 25;
  float initialFreq = 0.0f; // DC at rest; the schedule ramps it up
  float initialDuty[4] = {50, 50, 50, 50};
  float initialPhase[4] = {PHASES_CCW[0],
                                  PHASES_CCW[1],
                                  PHASES_CCW[2],
                                  PHASES_CCW[3]};

  JsonArray arr;
  if (doc.is<JsonArray>()) {
    arr = doc.as<JsonArray>();
  } else {
    JsonObject cfg = doc.as<JsonObject>();
    resolutionMs =
        cfg["resolution_ms"] | resolutionMs;
    initialFreq = cfg["initial_freq"] | initialFreq;

    JsonArray dutyArr = cfg["initial_duty"].as<JsonArray>();
    for (int i = 0; i < 4 && i < (int)dutyArr.size(); i++)
      initialDuty[i] = dutyArr[i] | initialDuty[i];

    // direction seeds all four phases; initial_phase overrides per channel.
    const char *dir = cfg["direction"] | "";
    if (strcasecmp(dir, "cw") == 0) {
      for (int i = 0; i < 4; i++)
        initialPhase[i] = PHASES_CW[i];
    }
    JsonArray phaseArr = cfg["initial_phase"].as<JsonArray>();
    for (int i = 0; i < 4 && i < (int)phaseArr.size(); i++)
      initialPhase[i] = phaseArr[i] | initialPhase[i];

    arr = cfg["schedule"].as<JsonArray>();
  }

  reserve(arr.size());
  _stepLabels.reserve(arr.size());

  // A trajectory point needs all four channels, so every one pushes a step.
  float curFreq = initialFreq;
  float curDuty[4];
  float curPhase[4];
  float curCarrier[4]; // NAN leaves that i's carrier untouched
  for (int i = 0; i < 4; i++) {
    curDuty[i] = initialDuty[i];
    curPhase[i] = initialPhase[i];
    curCarrier[i] = NAN;
  }

  // A label object sets the step label for the steps that follow it.
  String currentLabel;

  std::vector<String> unknownMethods;
  for (JsonObject obj : arr) {
    const char *methodCStr = obj["method"] | "";
    String method(methodCStr);
    int mask = obj["mask"] | 0;
    float value = obj["value"] | 0.0f;
    float from = obj["from"] | 0.0f;
    float to = obj["to"] | 0.0f;
    float shape = obj["shape"] | NAN; // NAN selects the ramp_mode default
    uint32_t durationMs = obj["duration_ms"] | 0;
    bool called = false;
    bool pushedTask = false;

    // "channels" is one channel or an array applied in one snapshot.
    int channels[4];
    int nChannels = 0;
    if (obj["channels"].is<JsonArray>()) {
      for (auto c : obj["channels"].as<JsonArray>()) {
        int ci = c.as<int>();
        if (ci >= 0 && ci < 4 && nChannels < 4)
          channels[nChannels++] = ci;
      }
    } else if (obj["channels"].is<int>()) {
      int ci = obj["channels"].as<int>();
      if (ci >= 0 && ci < 4)
        channels[nChannels++] = ci;
    }

    if (method == "addDutyCycleTask" && nChannels > 0) {
      for (int i = 0; i < nChannels; i++)
        curDuty[channels[i]] = constrain(value, 0.0f, 100.0f);
      addSequenceTask(makeTrajectoryTask(curFreq, curDuty,
                                    curPhase,
                                    curCarrier));
      called = true;
      pushedTask = true;
    } else if (method == "addPhaseTask" && nChannels > 0) {
      for (int i = 0; i < nChannels; i++)
        curPhase[channels[i]] = value;
      addSequenceTask(makeTrajectoryTask(curFreq, curDuty,
                                    curPhase,
                                    curCarrier));
      called = true;
      pushedTask = true;
    } else if (method == "addWaitTask") {
      addWaitTask(durationMs);
      called = true;
      pushedTask = true;
    } else if (method == "addLinearRampTask") {
      addRampTask(from, to, durationMs, TaskType::PWM_FREQ,
                 TaskMode::POLYNOMIAL, shape);
      curFreq = to;
      called = true;
      pushedTask = true;
    } else if (method == "addEaseRampTask") {
      addRampTask(from, to, durationMs, TaskType::PWM_FREQ,
                 TaskMode::EASE, shape);
      curFreq = to;
      called = true;
      pushedTask = true;
    } else if (method == "addExponentialRampTask") {
      addRampTask(from, to, durationMs, TaskType::PWM_FREQ,
                 TaskMode::EXPONENTIAL, shape);
      curFreq = to;
      called = true;
      pushedTask = true;
    } else if (method == "addCarrierRampTask") {
      addRampTask(from, to, durationMs, TaskType::CARRIER_DUTY,
                 TaskMode::POLYNOMIAL, shape);
      for (int i = 0; i < 4; i++)
        curCarrier[i] = to;
      called = true;
      pushedTask = true;
    } else if (method == "addCarrierEaseRampTask") {
      addRampTask(from, to, durationMs, TaskType::CARRIER_DUTY,
                 TaskMode::EASE, shape);
      for (int i = 0; i < 4; i++)
        curCarrier[i] = to;
      called = true;
      pushedTask = true;
    } else if (method == "addCarrierExponentialRampTask") {
      addRampTask(from, to, durationMs, TaskType::CARRIER_DUTY,
                 TaskMode::EXPONENTIAL, shape);
      for (int i = 0; i < 4; i++)
        curCarrier[i] = to;
      called = true;
      pushedTask = true;
    } else if (method == "addPhaseRampTask" && nChannels > 0) {
      // NAN leaves unnamed channels alone; same channel form as the setters.
      float starts[4] = {NAN, NAN, NAN, NAN};
      float ends[4] = {NAN, NAN, NAN, NAN};
      for (int i = 0; i < nChannels; i++) {
        starts[channels[i]] = from;
        ends[channels[i]] = to;
        curPhase[channels[i]] = to;
      }
      addRampTask(starts, ends, 4, durationMs,
                 TaskType::PWM_PHASE, TaskMode::EASE, shape);
      called = true;
      pushedTask = true;
    } else if (method == "addCarrierDutyCycleTask" && nChannels > 0) {
      for (int i = 0; i < nChannels; i++)
        curCarrier[channels[i]] = constrain(value, 0.0f, 100.0f);
      addSequenceTask(makeTrajectoryTask(curFreq, curDuty,
                                    curPhase,
                                    curCarrier));
      called = true;
      pushedTask = true;
    } else if (method == "setDirection") {
      // value 0 is clockwise, anything else counter-clockwise.
      const float *phases =
          (value != 0.0f) ? PHASES_CCW : PHASES_CW;
      for (int i = 0; i < 4; i++)
        curPhase[i] = phases[i];
      addSequenceTask(makeTrajectoryTask(curFreq, curDuty,
                                    curPhase,
                                    curCarrier));
      called = true;
      pushedTask = true;
    } else if (method == "activateChannels") {
      // A set mask bit drives the channel at value, a clear bit at zero.
      float onDuty = constrain(value, 0.0f, 100.0f);
      for (int i = 0; i < 4; i++)
        curCarrier[i] =
            ((mask >> i) & 1) ? onDuty : 0.0f;
      addSequenceTask(makeTrajectoryTask(curFreq, curDuty,
                                    curPhase,
                                    curCarrier));
      called = true;
      pushedTask = true;
    } else if (method == "label") {
      const char *labelCStr = obj["value"] | "";
      currentLabel = String(labelCStr);
      called = true; // recognized but appends nothing
    }

    if (pushedTask) {
      _stepLabels.push_back(currentLabel);
    }
    if (!called) {
      unknownMethods.push_back(method);
    }
  }

  if (!unknownMethods.empty()) {
    Serial.println("[JsonPwmSequencer] Unknown methods found in schedule:");
    for (const auto &m : unknownMethods) {
      Serial.println(m);
    }
  }

  compile(resolutionMs, initialFreq,
                    initialDuty, initialPhase);
  return true;
}
