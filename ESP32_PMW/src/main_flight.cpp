// Live PC-commanded flight: spin-up -> hover -> directional acceleration.
//
// `seq=` is the ONLY way to ramp the coils. The profile belongs to the host, where it
// can be retuned without a reflash; this firmware compiles in no ramp of its own.
#include "coil_probe.h"
#include "drive_common.h"
#include "SerialComm.h"
#include "constants.h"

static const float SPINUP_THROTTLE = 100.0f;
static const float COIL_AZ[NUM_CHANNELS] = {0.0f, 90.0f, 180.0f, 270.0f};

static const float MIX_GAIN = 0.6f;

#define SPIN_PHASES PHASES_CW

static PwmController ctl(PWM_PINS, SPIN_PHASES, INITIAL_DUTY, NUM_CHANNELS);
static PwmSequencer seq(&ctl);
static SerialComm comm;

enum State { IDLE, SPINUP, FLIGHT, LANDING, OFF }; // the host parses these numbers
static State state = IDLE;

static float collective = 0.0f; // % carrier ceiling (throttle)
static int seqN = 0;            // segments appended since the last seq=clear
static float seqF0 = 0.0f;      // start frequency of the first segment

static float azSet = 0.0f;      // deg
static float magSet = 0.0f;     // 0..1

// `duty=A:B:C:D`: per-channel carrier % that REPLACES the az/mag mixer while set. The
// host's tilt-servo loop (controller/control/tilt_servo.py) needs four independent
// amplitudes -- a balance trim plus one coil dropped to a fraction of its trimmed value --
// and the mixer's single cos lobe cannot express that. `collective` still scales it, so
// `throttle=` and the LANDING ramp behave the same either way. `duty=off` clears it.
static bool dutySet = false;
static float dutyPct[NUM_CHANNELS] = {100.0f, 100.0f, 100.0f, 100.0f};

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Thrust-vector mixer: drop the az-facing coils' ceilings so the disk tilts toward
// az. Strong side stays at collective, the balance reference. Verify sign on rig.
static void applyMixer() {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    float frac;
    if (dutySet) {
      frac = dutyPct[i] / 100.0f;
    } else {
      float drop = MIX_GAIN * magSet * max(0.0f, cosf((azSet - COIL_AZ[i]) * (float)DEG_TO_RAD));
      frac = 1.0f - drop;
    }
    ctl.setCarrierDutyCycle(i, clampf(collective * frac, 0.0f, 100.0f));
  }
}

static void splitFloats(const String &s, float *out, int n);

static bool cmdDuty(const String &arg) {
  if (arg == "off") { dutySet = false; return true; }
  float v[NUM_CHANNELS];
  for (int i = 0; i < NUM_CHANNELS; i++) v[i] = dutyPct[i];   // short command keeps the rest
  splitFloats(arg, v, NUM_CHANNELS);
  for (int i = 0; i < NUM_CHANNELS; i++) dutyPct[i] = clampf(v[i], 0.0f, 100.0f);
  dutySet = true;
  return true;
}

static void allCoilsOff() {
  for (int i = 0; i < NUM_CHANNELS; i++) ctl.setCarrierDutyCycle(i, 0.0f);
}

// Split "a:b:c" into up to `n` floats. Fields left unwritten keep the caller's default,
// so a short command is not a command full of zeros.
static void splitFloats(const String &s, float *out, int n) {
  String rest = s;
  for (int i = 0; i < n && rest.length(); i++) {
    int c = rest.indexOf(':');
    out[i] = (c < 0 ? rest : rest.substring(0, c)).toFloat();
    rest = (c < 0) ? String() : rest.substring(c + 1);
  }
}

// `seq=ramp:...` mode field, indexed by the integer the host sends. Same order as
// `TaskMode` and as `controller/control/constants.py` RAMP_POLYNOMIAL/EASE/EXPONENTIAL.
static const TaskMode RAMP_MODES[] = {
    TaskMode::POLYNOMIAL, TaskMode::EASE, TaskMode::EXPONENTIAL};
static const int N_RAMP_MODES = sizeof(RAMP_MODES) / sizeof(RAMP_MODES[0]);

// Each handler returns true when the command took effect, false when it declined and has
// already said why. Only a true prints the state echo.
static bool cmdSeq(const String &arg) {
  if (state != IDLE) { Serial.printf("!seq state=%d\n", (int)state); return false; }

  const int colon = arg.indexOf(':');
  const String sub = (colon < 0) ? arg : arg.substring(0, colon);
  const String rest = (colon < 0) ? String() : arg.substring(colon + 1);

  if (sub == "clear") {
    seq.clear();
    seqN = 0;
    Serial.println("seq cleared");
    return true;
  }
  if (sub == "go") {
    if (seqN == 0) { Serial.println("!seq empty"); return false; }
    seq.compile(25, seqF0, INITIAL_DUTY, SPIN_PHASES);
    collective = SPINUP_THROTTLE;
    seq.start();
    state = SPINUP;
    Serial.printf("seq go: %d segment(s) from %.2f Hz\n", seqN, seqF0);
    return true;
  }
  if (sub == "ramp") {
    float v[5] = {0, 0, 0, 0, 1.0f};   // from, to, ms, mode, k
    splitFloats(rest, v, 5);
    if (!(v[0] > 0.0f) || !(v[1] > 0.0f) || !(v[2] > 0.0f)) {
      Serial.println("!seq ramp needs from:to:ms>0");
      return false;
    }
    int mi = (int)(v[3] + 0.5f);
    if (mi < 0 || mi >= N_RAMP_MODES) {
      Serial.printf("!seq ramp mode %d: use 0 poly, 1 ease, 2 exp\n", mi);
      return false;
    }
    if (seqN == 0) seqF0 = v[0];
    seq.addRampTask(v[0], v[1], (uint32_t)v[2], TaskType::PWM_FREQ, RAMP_MODES[mi], v[4]);
    seqN++;
    Serial.printf("seq+ %d: %.2f -> %.2f Hz over %.0f ms mode %d k %.2f\n",
                  seqN, v[0], v[1], v[2], mi, v[4]);
    return true;
  }
  Serial.println("? seq=clear|ramp:from:to:ms:mode:k|go");
  return false;
}

// probe=<hz>[:<ms>[:<carrier %>]] -- one lock-in burst measuring the coil CURRENT phase
// per channel. IDLE only: it energises, and it must not be able to fire mid-flight.
//
// The burst BLOCKS, so for its duration neither the GPIO14 button nor a host `stop` is
// serviced. That is why PROBE_MAX_MS is short and the coils are cut the instant it
// returns: the window is bounded by construction rather than by anything watching it.
// Nothing here holds the coils on after the call.
static const uint32_t PROBE_MAX_MS = 2000;
static const float PROBE_DUTY = 60.0f;   // % carrier. Near resonance full drive pulled
                                         // 4.08 A (theory.md 18.8) against a 2 A
                                         // continuous rating, and the lock-in does not
                                         // need the amplitude -- only the angle.
static const uint32_t PROBE_SETTLE_MS = 250;   // >> the RLC ringdown, Q/f0 is ~ms here

static bool cmdProbe(const String &arg) {
  if (state != IDLE) { Serial.printf("!probe state=%d\n", (int)state); return false; }

  float v[3] = {0.0f, 1200.0f, PROBE_DUTY};   // hz, ms, carrier %
  splitFloats(arg, v, 3);
  if (!(v[0] > 0.0f)) { Serial.println("!probe needs hz>0"); return false; }
  const uint32_t ms = (uint32_t)clampf(v[1], 100.0f, (float)PROBE_MAX_MS);
  const float duty = clampf(v[2], 0.0f, 100.0f);

  ctl.setGlobalFrequency(v[0]);
  for (int i = 0; i < NUM_CHANNELS; i++) ctl.setCarrierDutyCycle(i, duty);
  delay(PROBE_SETTLE_MS);

  ProbeResult r;
  // SPIN_PHASES is the commanded reference the measured angle is taken against. Note it
  // is the BASE phase: any trim already applied is what we are trying to see the effect
  // of, so a post-trim probe correctly reports the RESIDUAL.
  const bool ok = coilProbe(ADC_PINS, NUM_CHANNELS, v[0], ms, SPIN_PHASES, r);
  allCoilsOff();
  ctl.setGlobalFrequency(0.0f);   // back to DC idle, so the echo does not read as driving

  if (!ok) { Serial.println("!probe no samples"); return false; }
  Serial.printf("PROBE f=%.2f A=%.0f,%.2f B=%.0f,%.2f C=%.0f,%.2f D=%.0f,%.2f "
                "n=%lu coh=%.2f\n",
                v[0], r.amp[0], r.phaseDeg[0], r.amp[1], r.phaseDeg[1],
                r.amp[2], r.phaseDeg[2], r.amp[3], r.phaseDeg[3],
                (unsigned long)r.n, r.coherence);
  return true;
}

static bool cmdFreq(const String &arg) {
  float hz = arg.toFloat();
  if (state != FLIGHT) { Serial.printf("!freq state=%d\n", (int)state); return false; }
  if (hz <= 0.0f) { Serial.printf("!freq=%.2f\n", hz); return false; }
  ctl.setGlobalFrequency(hz);
  return true;
}

// Commands are `key` or `key=value`. Splitting on '=' once is what keeps the arms free of
// hand-counted substring offsets, which had to match each prefix's length by eye.
static void dispatch(String cmd) {
  cmd.trim();
  cmd.toLowerCase();

  int eq = cmd.indexOf('=');
  const String key = (eq < 0) ? cmd : cmd.substring(0, eq);
  const String arg = (eq < 0) ? String() : cmd.substring(eq + 1);

  bool ok;
  if (key == "throttle")   { collective = clampf(arg.toFloat(), 0.0f, 100.0f); ok = true; }
  else if (key == "az")    { azSet = arg.toFloat();                            ok = true; }
  else if (key == "mag")   { magSet = clampf(arg.toFloat(), 0.0f, 1.0f);       ok = true; }
  else if (key == "hover") { magSet = 0.0f;                                    ok = true; }
  else if (key == "land")  { if (state == SPINUP || state == FLIGHT) state = LANDING;
                             ok = true; }
  else if (key == "stop")  { allCoilsOff(); state = OFF;                       ok = true; }
  else if (key == "freq")   ok = cmdFreq(arg);
  else if (key == "seq")    ok = cmdSeq(arg);
  else if (key == "probe")  ok = cmdProbe(arg);
  else if (key == "duty")   ok = cmdDuty(arg);
  else {
    Serial.printf("? '%s' (seq=|throttle=|az=|mag=|duty=|hover|land|stop|freq=|probe=)\n",
                  cmd.c_str());
    return;
  }
  if (!ok) return;

  Serial.printf("state=%d col=%.0f az=%.0f mag=%.2f freq=%.2f duty=%s\n",
                (int)state, collective, azSet, magSet, ctl.getFrequency(),
                dutySet ? "set" : "off");
}

void setup() {
  driveBoot();
  ctl.begin(); // DC; the ramp sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS, /*tripA*/ 0.0f);
  // No-op until COIL_F0_HZ / COIL_Q are filled in from a probe sweep: setPhaseTrim
  // refuses a table with any non-positive entry, so an uncalibrated build drives the
  // commanded phase raw rather than a guessed correction.
  ctl.setPhaseTrim(COIL_F0_HZ, COIL_Q);
  Serial.printf("[flight] phase trim %s\n",
                ctl.phaseTrimActive() ? "ARMED from drive_common.h" : "off (uncalibrated)");
}

void loop() {
  String line = comm.handleSerialComm();
  if (line.length()) dispatch(line);

  ctl.run(); // sense + balance + overcurrent trip

  switch (state) {
    case SPINUP:
      seq.run();
      applyMixer();
      if (seq.isDone()) { state = FLIGHT; Serial.println("state=2 (FLIGHT)"); }
      break;
    case FLIGHT:
      applyMixer();
      break;
    case LANDING: {
      static unsigned long lastStep = 0;
      if (millis() - lastStep >= 20) {
        lastStep = millis();
        if (ctl.rampDownStep(2.0f)) { state = OFF; Serial.println("state=4 (OFF)"); }
      }
      break;
    }
    case IDLE:
    case OFF:
      allCoilsOff();
      break;
  }

  driveTelemetry(ctl);
}
