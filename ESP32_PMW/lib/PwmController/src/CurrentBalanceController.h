#pragma once

#include <Arduino.h>

struct BalanceConfig {
  float kp = 2.2f;
  float ki = 0.10f;
  float kd = 0.15f;
  float dutyMin = 5.0f;
  float dutyMax = 100.0f;
  float iMax = 12.0f;
  float overcurrentBackoffPct = 5.0f;
  float nominalTickMs = 2.0f;
  // hysteresis: stops noise flapping the latched anchor.
  float minSwitchMarginA = 0.3f;
  float minRampPctPerMs = 0.05f;

  float minSignalA = 0.25f;

  float refBandPct = 0.5f;
};

class CurrentBalanceController {
public:
  static const int N = 4;

  explicit CurrentBalanceController(const BalanceConfig &cfg = BalanceConfig());

  void reset(float startDuty);

  void step(const float *iMeas, float dtMs, const float *ceiling,
              float *dutyOut);

  void setGains(float kp, float ki, float kd) {
    _cfg.kp = kp;
    _cfg.ki = ki;
    _cfg.kd = kd;
  }
  void setRamp(float pctPerMs) { _cfg.minRampPctPerMs = pctPerMs; }

private:
  BalanceConfig _cfg;
  float _integrator[N];
  float _dutyOut[N];
  float _lastErr[N];
  int _idxMin;
  bool _holdFrozen;
  float _holdTarget;
};
