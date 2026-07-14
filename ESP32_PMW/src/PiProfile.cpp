#include "PiProfile.h"

#include <ArduinoJson.h>
#include <FS.h>
#include <SPIFFS.h>
#include <string.h>

namespace {
const char *CHANNEL_NAMES[kNumChannels] = {"A", "B", "C", "D"};
}

bool loadPiProfile(const char *path, RatioCurrentController::Config *out) {
  File file = SPIFFS.open(path, "r");
  if (!file) return false;
  size_t size = file.size();
  std::unique_ptr<char[]> buf(new char[size + 1]);
  file.readBytes(buf.get(), size);
  buf[size] = '\0';
  file.close();

  JsonDocument doc;
  auto err = deserializeJson(doc, buf.get());
  if (err) {
    Serial.printf("[PiProfile] parse failed: %s (file=%u bytes)\n", err.c_str(),
                  (unsigned)size);
    return false;
  }

  const char *mode = doc["mode"] | "";
  RatioCurrentController::Config cfg;
  if (strcmp(mode, "shared_constraint") == 0) {
    cfg.sharedConstraint = true;
  } else if (strcmp(mode, "independent") == 0) {
    cfg.sharedConstraint = false;
  } else {
    Serial.printf("[PiProfile] missing/unknown \"mode\" (\"%s\") -- must be "
                  "\"shared_constraint\" or \"independent\"\n", mode);
    return false;
  }

  JsonObject ratios = doc["ratios"];
  if (ratios.isNull()) {
    Serial.println("[PiProfile] missing \"ratios\" object");
    return false;
  }
  for (int i = 0; i < kNumChannels; i++) {
    if (!ratios[CHANNEL_NAMES[i]].is<float>()) {
      Serial.printf("[PiProfile] \"ratios\" missing channel \"%s\"\n", CHANNEL_NAMES[i]);
      return false;
    }
    cfg.ratios[i] = ratios[CHANNEL_NAMES[i]].as<float>();
    if (cfg.ratios[i] <= 0.0f) {
      Serial.printf("[PiProfile] \"ratios.%s\"=%.3f must be > 0\n", CHANNEL_NAMES[i],
                    cfg.ratios[i]);
      return false;
    }
  }

  JsonObject gains = doc["gains"];
  cfg.kp = gains["kp"] | 2.2f;
  cfg.ki = gains["ki"] | 0.10f;
  cfg.kd = gains["kd"] | 0.15f;

  // Defaults match main_current_pid.cpp's converged/documented values.
  cfg.rampPctPerMs = doc["ramp_pct_per_ms"] | 0.05f;
  cfg.dutyMin = doc["duty_min"] | 5.0f;
  cfg.dutyMax = doc["duty_max"] | 100.0f;
  cfg.iMaxA = doc["i_max_a"] | 12.0f;
  cfg.overcurrentBackoffPct = doc["overcurrent_backoff_pct"] | 5.0f;
  cfg.minSwitchMarginA = doc["min_switch_margin_a"] | 0.3f;
  cfg.nominalTickMs = doc["nominal_tick_ms"] | 2.0f;
  // Independent mode only -- see RatioCurrentController.h's header comment
  // for why this exists (prevents the magnitude ramp from outrunning the
  // PI loop's actual convergence). Untuned placeholder, matches this
  // project's precedent for rate constants (e.g. MIN_RAMP_PCT_PER_MS).
  cfg.magnitudeSettleTolA = doc["magnitude_settle_tol_a"] | 0.2f;

  *out = cfg;
  return true;
}
