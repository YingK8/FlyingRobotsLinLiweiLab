// Self-contained, autonomous tilt experiment + PI compensation -- no JSON,
// no SPIFFS, nothing loaded at runtime, no serial command listener at all.
// Upload and it runs: ARMING (fixed delay, self-calibrate ADC zero) ->
// RUNNING immediately, straight through to DONE with no way to interrupt
// it early except the hard overcurrent trip or cutting power -- deliberate,
// see the plan discussion. Use tools/record_serial.py to passively observe
// (it never writes to the port either). This is a SKELETON -- the tilt
// schedule (buildTiltSchedule below) and PI ratios/gains (PI_CFG below) are
// starting points ported directly from task_sequences/tilt.json +
// task_sequences/pi_profile_tilt.json; edit either in plain C++, no JSON
// round-trip required.

#include <Arduino.h>
#include <math.h>

#include "ExperimentPhase.h"
#include "PWMController.h"
#include "PWMSequencer.h"
#include "RatioCurrentController.h"
#include "constants.h"
#include "CurrentSense.h"
#include "safety_startup.h"
#include "telemetry.h"

// ============================== DRIVE ==============================
// tilt.json: setDirection value=1 -> CCW.
const float PHASES_CCW[NUM_CHANNELS] = {90.0, 270.0, 180.0, 0.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};  // commutation duty, unchanged throughout
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {0, 0, 0, 0};
const float DRIVE_FREQ = 190.0f;   // controller->begin()/compile() baseline -- see main_experiment.cpp's note
const float START_FREQ = 1.0f;   // NOT 0.0f -- setGlobalFrequency() divides by
                                  // newHz, so a literal 0Hz corrupts PWM timing
const float END_FREQ = 200.0f;
const unsigned long RAMP_DURATION_MS = 30000;  // tilt.json: addEaseRampTask 1->210Hz over 30s
const unsigned long HOLD_MS = 2500;            // tilt.json: addWaitTask per hold step

// Full-scale balanced current setpoint, in AMPS. This is the single knob that
// "generalises to different current levels": the PI reference each tick is
// (scheduled carrier % / 100) * I_TARGET_A, so at 100% carrier every channel
// is regulated to I_TARGET_A and the schedule's 100->0% carrier steps sweep
// the balanced current from I_TARGET_A down to 0 in 10% steps. With balanced
// ratios [1,1,1,1] the achievable balanced current is capped by the WEAKEST
// channel (channel C, ~1.55 A at 200 Hz full duty on this rig -- see
// data/2026-07-14_tilt run3); set above that and C simply pins at 100% duty
// while the others still regulate. Change this value to run the same sweep at
// a different current level.
//
// 2026-07-14: pushed to 5.0 A for a max-performance run at 15 V -- deliberately
// ABOVE what the weakest channel (C) can follow, so C pins at 100% duty and the
// balanced current settles at C's actual 15 V ceiling (the run reveals it). The
// firmware overcurrent trip has been REMOVED for this (user request) -- the
// PSU's own current limit is now the ONLY hardware protection.
const float I_TARGET_A = 5.0f;

// ============================== PI COMPENSATION ==============================
// Independent-mode PI with EQUAL ratios [1,1,1,1] -> pure balance: every
// channel PI-tracks the SAME current target (this tick's reference, in amps),
// so the weakest channel is driven to high duty and the strongest backs off.
// The reference is supplied by the caller each tick (see loop()) as
// (carrier% / 100) * I_TARGET_A -- computeTickWithReference() only, no internal
// magnitude governor. rampPctPerMs / minSwitchMarginA / magnitudeSettleTolA
// are unused by computeTickWithReference (they belong to computeTick's governor
// and the shared-constraint anchor, neither of which this firmware calls).
RatioCurrentController::Config PI_CFG = {
  {1.0f, 1.0f, 1.0f, 1.0f},  // ratios (A, B, C, D) -- equal => balance
  false,                      // sharedConstraint -- independent mode
  2.2f, 0.10f, 0.15f,         // kp, ki, kd
  0.05f,                      // rampPctPerMs (unused by computeTickWithReference)
  0.3f,                       // minSwitchMarginA (shared-constraint mode only, unused here)
  5.0f, 100.0f,                // dutyMin, dutyMax
  12.0f,                       // iMaxA (RatioCurrentController's own soft backoff)
  5.0f,                        // overcurrentBackoffPct
  2.0f,                        // nominalTickMs
  0.2f,                        // magnitudeSettleTolA (unused by computeTickWithReference)
};

// ============================== CURRENT SENSE ==============================
CurrentSense currentSense(ADC_PINS, SENS, NUM_CHANNELS);

// Hard overcurrent trip: unconditional latch-off if any channel exceeds this.
// Set to 10 A to match the PSU (ASU) current limit -- the firmware trips at the
// same ceiling the supply would, so a runaway latches off in firmware rather
// than relying solely on the PSU folding back. With no serial e-stop, this and
// physically cutting power are the only ways this firmware stops early.
const float I_SAFETY_MAX_A = 10.0f;

PWMController *controller;
PWMSequencer *seq;
RatioCurrentController *ratioController;

enum Phase { ARMING, RUNNING, DONE };
Phase phase = ARMING;
unsigned long phase_start = 0;

void allCoilsOff() {
  for (int i = 0; i < NUM_CHANNELS; i++) controller->setCarrierDutyCycle(i, 0.0f);
}

// Builds the tilt duty schedule directly against PWMSequencer's primitives
// (ported from task_sequences/tilt.json): CCW, 1->210Hz ease ramp, then
// 100%->0% carrier duty in 10% steps (2.5s holds), all channels equal --
// commutation duty stays fixed at INITIAL_DUTY_CYCLES throughout, only
// carrier duty is scheduled here.
void buildTiltSchedule() {
  // Command full carrier duty BEFORE the frequency ramp starts. A PWM_FREQ
  // ramp task never touches carrier duty (see PWMSequencer::step()), and
  // _currentCarrierDutyCycles starts out NAN -- without this,
  // scheduledCarrierDutyCycle() stays NAN through the whole ramp, the PI
  // compensation block in loop() is skipped (isnan(reference) guard), and
  // the coils sit at their initial 0% duty the entire time instead of
  // running at full current. This instant (duration=0) CARRIER_DUTY task
  // sets the reference to 100 and it then carries forward untouched through
  // the frequency-only ramp that follows.
  const float fullCarrier[NUM_CHANNELS] = {100.0f, 100.0f, 100.0f, 100.0f};
  seq->addRampTask(fullCarrier, fullCarrier, NUM_CHANNELS, 0, TaskType::CARRIER_DUTY);
  seq->addRampTask(START_FREQ, END_FREQ, RAMP_DURATION_MS, TaskType::PWM_FREQ, TaskMode::EASE);
  for (int pct = 100; pct >= 0; pct -= 10) {
    float carrier[NUM_CHANNELS] = {(float)pct, (float)pct, (float)pct, (float)pct};
    seq->addSequenceTask(makeTrajectoryTask(END_FREQ, INITIAL_DUTY_CYCLES, PHASES_CCW, carrier));
    seq->addWaitTask(HOLD_MS);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // SAFE STARTUP: gates forced LOW before constructing anything.
  forceAllGatesLow();
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  currentSense.seed();

  controller = new PWMController(PWM_PINS, PHASES_CCW, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  // Explicit non-zero starting frequency -- begin(0.0f) divides by zero
  // inside setGlobalFrequency() and permanently corrupts commutation timing.
  controller->begin(DRIVE_FREQ);
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  allCoilsOff();  // explicit: carriers latched at 0 after LEDC attach

  seq = new PWMSequencer(controller);
  buildTiltSchedule();
  seq->compile(25, DRIVE_FREQ, INITIAL_DUTY_CYCLES, PHASES_CCW);

  ratioController = new RatioCurrentController(PI_CFG);

  phase = ARMING;
  phase_start = millis();
  Serial.println("main_tilt_pi: arming 3s (coils confirmed off, ADC zero "
                  "recalibrating), then autonomously running the tilt "
                  "schedule with PI compensation -- NO e-stop listener; a 10A "
                  "overcurrent trip or cutting power are the only ways it stops "
                  "early");
}

void loop() {
  controller->step();
  unsigned long now = millis();

  // ADC sampling paced separately from everything else -- see
  // CurrentSense.cpp for why (ESP32 ADC mux/S&H settling requirement).
  static unsigned long last_adc_us = micros();
  unsigned long now_us = micros();
  const float ADC_SAMPLE_MS = 1.0f;
  float dt_adc_ms = (float)(now_us - last_adc_us) / 1000.0f;
  if (dt_adc_ms >= ADC_SAMPLE_MS) {
    currentSense.update(dt_adc_ms);
    last_adc_us = now_us;
  }

  // dt for the PI compensation tick (KI/KD rate-scaling).
  static unsigned long last_control_us = now_us;
  float control_dt_ms = (float)(now_us - last_control_us) / 1000.0f;
  if (control_dt_ms <= 0.0f) control_dt_ms = 0.001f;  // guard div-by-zero only
  last_control_us = now_us;

  switch (phase) {
  case ARMING:
    if (ExperimentPhase::armingTick(now, phase_start, ARM_MS, controller,
                                     NUM_CHANNELS, currentSense, LED_PIN)) {
      ratioController->reset();
      seq->start();
      phase = RUNNING;
      Serial.println("ARMED -> running tilt schedule");
    }
    break;

  case RUNNING: {
    seq->step();

    // PI compensation, always on. The reference is a CURRENT setpoint in
    // AMPS: the schedule's OWN commanded carrier duty (0-100%, the
    // sequencer's intended state -- not PWMController's actuator register,
    // see PWMSequencer::scheduledCarrierDutyCycle()'s comment) as a FRACTION
    // times I_TARGET_A. pidStep() compares target against i_meas in amps, so
    // the reference MUST be amps -- feeding the raw 0-100 duty percent makes
    // target=100A (unreachable) and pins every channel at dutyMax (confirmed
    // on hardware, run1/run2). With balanced ratios [1,1,1,1] each channel
    // independently tracks the same reference => currents equalise. NAN
    // before the schedule commands any carrier duty -- skip until it does.
    float carrierPct = seq->scheduledCarrierDutyCycle(0);
    if (!isnan(carrierPct)) {
      float reference = (carrierPct / 100.0f) * I_TARGET_A;  // amps
      float duty_out[NUM_CHANNELS];
      ratioController->computeTickWithReference(currentSense.i_meas, control_dt_ms,
                                                reference, duty_out);
      for (int i = 0; i < NUM_CHANNELS; i++)
        controller->setCarrierDutyCycle(i, duty_out[i]);
    }

    for (int i = 0; i < NUM_CHANNELS; i++) {
      if (currentSense.i_meas[i] > I_SAFETY_MAX_A) {
        allCoilsOff();
        phase = DONE;
        Serial.printf("SAFETY: channel %d overcurrent (%.2fA) -- latching off\n",
                      i, currentSense.i_meas[i]);
        break;
      }
    }
    if (phase != RUNNING) break;

    static unsigned long last_periodic_ms = 0;
    const unsigned long PERIODIC_TELEMETRY_MS = 200;
    if (now - last_periodic_ms >= PERIODIC_TELEMETRY_MS) {
      last_periodic_ms = now;
      float dutyPct[NUM_CHANNELS];
      for (int i = 0; i < NUM_CHANNELS; i++)
        dutyPct[i] = controller->getCarrierDutyCycle(i);
      float carrierPct = seq->scheduledCarrierDutyCycle(0);
      float ref = isnan(carrierPct) ? NAN : (carrierPct / 100.0f) * I_TARGET_A;
      Serial.printf("t=%lu | carrier=%.0f%% ref=%.2fA | ", now,
                    isnan(carrierPct) ? 0.0f : carrierPct, ref);
      printCurrentAndDuty(currentSense.i_meas, dutyPct);
      Serial.println();
    }

    if (seq->isDone()) {
      allCoilsOff();
      phase = DONE;
      Serial.println("tilt schedule complete -> DONE (coils latched OFF)");
    }
    break;
  }

  case DONE:
    // SAFE SHUTDOWN: coils latched OFF permanently; slow heartbeat so the
    // board still reads as alive without ever re-energizing.
    ExperimentPhase::latchedOffTick(now, controller, NUM_CHANNELS, LED_PIN);
    break;
  }
}
