#include "CoilBalancer.h"

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

CoilBalancer::CoilBalancer(PhaseController* controller,
                           const CoilBalancerConfig& cfg)
    : _ctrl(controller), _cfg(cfg) {}

void CoilBalancer::begin() {
  // 12-bit ADC, 11 dB attenuation -> usable ~0..3.1 V (CS node must stay <3.3 V).
  analogReadResolution(12);
  for (int ch = 0; ch < 4; ch++) {
    analogSetPinAttenuation(_cfg.adcPins[ch], ADC_11db);
  }
  reset();
}

void CoilBalancer::setEnabled(bool en) {
  if (en && !_enabled) {
    // Re-arming: start from a clean slate at the ceiling so we only ever trim DOWN
    // toward balance, never step a coil up unexpectedly.
    reset();
  }
  _enabled = en;
}

void CoilBalancer::reset() {
  _trim[0] = _trim[1] = 0.0f;
  for (int ch = 0; ch < 4; ch++) {
    _dutyApplied[ch] = _cfg.ceilingDuty[ch];
    _ctrl->setCarrierDutyCycle(ch, _cfg.ceilingDuty[ch]);
  }
  _lastUpdateMs = millis();
}

// Oversampled CS read -> current proxy (mV / shunt_ohms). analogReadMilliVolts
// uses the chip's factory ADC calibration, so channels are comparable in mV.
float CoilBalancer::readCurrent(int ch) {
  uint32_t acc = 0;
  uint16_t n = _cfg.adcOversample > 0 ? _cfg.adcOversample : 1;
  for (uint16_t i = 0; i < n; i++) {
    acc += analogReadMilliVolts(_cfg.adcPins[ch]);
  }
  float mV = (float)acc / (float)n;
  float r = _cfg.shuntOhms[ch] > 0.0f ? _cfg.shuntOhms[ch] : 1.0f;
  return mV / r;  // proxy current (mA if mV/ohm); consistent scale across channels
}

void CoilBalancer::update() {
  unsigned long now = millis();
  if (now - _lastUpdateMs < _cfg.updatePeriodMs) return;
  float dt = (now - _lastUpdateMs) * 0.001f;
  _lastUpdateMs = now;

  // Always sample (telemetry stays live even when not actuating).
  for (int ch = 0; ch < 4; ch++) _current[ch] = readCurrent(ch);

  if (!_enabled) return;
  if (dt <= 0.0f) return;

  applyPair(0, dt);
  applyPair(1, dt);
}

// One opposing pair (p, q). Signed convention: err > 0  =>  p is the STRONGER
// coil. The PI output `trim` is applied asymmetrically so that ONLY the stronger
// coil is pulled below its ceiling; the weaker coil stays at full drive, which is
// what maximises min(I_p, I_q) while the loop drives |err| -> 0.
void CoilBalancer::applyPair(int pairIdx, float dt) {
  int p = _cfg.pairs[pairIdx][0];
  int q = _cfg.pairs[pairIdx][1];

  float err = _current[p] - _current[q];   // + => p stronger
  _pairErr[pairIdx] = err;

  // Integrate only outside the deadband so ADC noise near balance doesn't make
  // the trim wander (which would dither the coils audibly/thermally).
  if (fabsf(err) > _cfg.deadband) {
    _trim[pairIdx] += _cfg.ki * err * dt;
    _trim[pairIdx] = clampf(_trim[pairIdx], -_cfg.trimMax, _cfg.trimMax);
  }

  // PI command. Proportional term is bounded to the same trim envelope so a
  // transient can't command more trim than trimMax.
  float cmd = clampf(_trim[pairIdx] + _cfg.kp * err, -_cfg.trimMax, _cfg.trimMax);

  // cmd > 0  -> trim p (the stronger one); q held at ceiling. And vice-versa.
  float dutyP = _cfg.ceilingDuty[p] - fmaxf(0.0f,  cmd);
  float dutyQ = _cfg.ceilingDuty[q] - fmaxf(0.0f, -cmd);

  dutyP = clampf(dutyP, _cfg.dutyFloor, _cfg.ceilingDuty[p]);
  dutyQ = clampf(dutyQ, _cfg.dutyFloor, _cfg.ceilingDuty[q]);

  _dutyApplied[p] = dutyP;
  _dutyApplied[q] = dutyQ;
  _ctrl->setCarrierDutyCycle(p, dutyP);
  _ctrl->setCarrierDutyCycle(q, dutyQ);
}
