#pragma once

#include <Arduino.h>
#include "PhaseController.h"

// ============================================================================
//  CoilBalancer  --  low-level current-balance feedback controller.
//
//  GOAL (per opposing pair): make the two coils carry the SAME current while
//  keeping that shared current as HIGH as possible. Coil current ~ ampere-turns
//  ~ magnetic field, so equal current => balanced opposing fields => no net
//  tilt/translation from coil mismatch.
//
//  Objective, made precise:  for a pair (p, q) we want to
//      maximise   min(I_p, I_q)        (don't waste drive: keep the weak one full)
//      minimise   |I_p - I_q|          (balance the pair)
//  The optimum is: pin the WEAKER coil at its carrier ceiling (max drive, so the
//  min current is as large as the hardware allows) and TRIM the STRONGER coil's
//  carrier duty down until its current matches. A single signed PI integrator per
//  pair does exactly this -- see update().
//
//  We can only ever reduce a coil's drive (carrier duty <= ceiling, and 100% =
//  constant-on per PhaseController). So balancing is always "trim the strong one
//  down to the weak one", never "boost the weak one" -- which is why holding the
//  weak coil at the ceiling is what maximises the achievable common current.
//
//  Measurement: each channel has a current-sense (CS) node feeding an ESP32 ADC
//  pin through a per-channel shunt. current_proxy = adc_mV / shunt_ohms. Units are
//  arbitrary-but-consistent across channels (the per-channel shunt normalises the
//  small resistor mismatch so A and B are directly comparable).
//
//  HARDWARE NOTE (classic ESP32 / ESP-WROOM-32): only ADC1 (GPIO 32-39) and ADC2
//  (0,2,4,12-15,25-27) can do analogRead. GPIO 5/18/19/21 have NO ADC. The four
//  free input-only ADC1 pins 34/35/36/39 are the correct CS pins on this board.
// ============================================================================

struct CoilBalancerConfig {
  // --- per-channel CS measurement (index order A=0, B=1, C=2, D=3) ----------
  int   adcPins[4];     // ESP32 ADC pin per channel (must be ADC-capable!)
  float shuntOhms[4];   // CS shunt per channel; current_proxy = mV / ohms

  // --- actuator limits (per channel) ----------------------------------------
  float ceilingDuty[4]; // carrier duty the WEAK coil is held at (max drive, %)
  float dutyFloor;      // never trim any coil below this carrier duty (%)
  float trimMax;        // max trim below ceiling the controller may apply (%)

  // --- which coils oppose each other ----------------------------------------
  // Two pairs of channel indices. Default {{0,1},{2,3}} = (A,B) and (C,D).
  int pairs[2][2];

  // --- PI gains (current-error -> trim%). err is in current-proxy units. -----
  float kp;             // proportional gain  [duty% per current unit]
  float ki;             // integral gain      [duty% per (current unit * s)]
  float deadband;       // |err| below this freezes the integrator (anti-dither)

  // --- timing / filtering ---------------------------------------------------
  uint32_t updatePeriodMs;  // control update period (loop is rate-limited to this)
  uint16_t adcOversample;   // ADC samples averaged per channel per update
};

class CoilBalancer {
public:
  // The balancer drives carrier duties through an already-begun PhaseController.
  CoilBalancer(PhaseController* controller, const CoilBalancerConfig& cfg);

  // Configure ADC (resolution + attenuation) and reset state. Call once in setup
  // AFTER controller->initCarrierPWM(...).
  void begin();

  // Call frequently from loop(). Internally rate-limited to cfg.updatePeriodMs.
  // When disabled it still samples currents (for telemetry) but does not actuate.
  void update();

  // Stop balancing and return every coil to its ceiling duty (full drive).
  void reset();

  void setEnabled(bool en);
  bool isEnabled() const { return _enabled; }

  // --- telemetry ------------------------------------------------------------
  float getCurrent(int ch)    const { return _current[ch]; }   // proxy units
  float getDutyApplied(int ch)const { return _dutyApplied[ch]; } // % carrier duty
  float getPairError(int p)   const { return _pairErr[p]; }     // I_p0 - I_p1
  float getPairTrim(int p)    const { return _trim[p]; }        // signed trim %

private:
  float readCurrent(int ch);   // oversampled mV / shunt -> proxy current
  void  applyPair(int pairIdx, float dt);

  PhaseController* _ctrl;
  CoilBalancerConfig _cfg;

  bool  _enabled = false;
  unsigned long _lastUpdateMs = 0;

  float _current[4]      = {0, 0, 0, 0};
  float _dutyApplied[4]  = {0, 0, 0, 0};
  float _trim[2]         = {0, 0};   // signed integrator state per pair
  float _pairErr[2]      = {0, 0};
};
