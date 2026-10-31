#include "coil_probe.h"
#include "drive_common.h"
#include "frame_link.h"
#include "constants.h"

static const float SPINUP_THROTTLE = 100.0f;
static const float COIL_AZ[NUM_CHANNELS] = {0.0f, 90.0f, 180.0f, 270.0f};

static const float MIX_GAIN = 0.6f;

#define SPIN_PHASES PHASES_CW

static PwmController ctl(PWM_PINS, SPIN_PHASES, INITIAL_DUTY, NUM_CHANNELS);
static PwmSequencer seq(&ctl);
static drive_frame::Framer framer;

static float phaseDeg[NUM_CHANNELS] = {SPIN_PHASES[0], SPIN_PHASES[1],
                                       SPIN_PHASES[2], SPIN_PHASES[3]};

static const uint16_t ACK_EVERY = 20;
static uint16_t lastDriveSeq = 0;
static uint16_t sinceAck = 0;

enum State { IDLE, SPINUP, FLIGHT, LANDING, OFF };
static State state = IDLE;

static float collective = 0.0f;
static int seqN = 0;
static float seqF0 = 0.0f;

static float azSet = 0.0f;
static float magSet = 0.0f;

// `duty=` replaces the tilt mixer while set; `throttle=` still scales it
static bool dutySet = false;
static float dutyPct[NUM_CHANNELS] = {100.0f, 100.0f, 100.0f, 100.0f};

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// verify the tilt sign on the rig before trusting it
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

// not a drive path: changing a phase energises nothing
static bool cmdDuty(const String &arg) {
  if (arg == "off") { dutySet = false; return true; }
  float v[NUM_CHANNELS];
  for (int i = 0; i < NUM_CHANNELS; i++) v[i] = dutyPct[i];
  splitFloats(arg, v, NUM_CHANNELS);
  for (int i = 0; i < NUM_CHANNELS; i++) dutyPct[i] = clampf(v[i], 0.0f, 100.0f);
  dutySet = true;
  return true;
}

// IDLE only: clearing mid-run empties the queue and jumps to FLIGHT still energised
static bool cmdPhase(const String &arg) {
  float v[NUM_CHANNELS];
  for (int i = 0; i < NUM_CHANNELS; i++) v[i] = phaseDeg[i];
  splitFloats(arg, v, NUM_CHANNELS);
  for (int i = 0; i < NUM_CHANNELS; i++) phaseDeg[i] = v[i];
  ctl.setPhases(phaseDeg);
  return true;
}

static void allCoilsOff() {
  for (int i = 0; i < NUM_CHANNELS; i++) ctl.setCarrierDutyCycle(i, 0.0f);
}

static void reportStop() {
  const int64_t t0 = esp_timer_get_time();
  led().sendStopSignal();
  const int64_t t1 = esp_timer_get_time();
  Serial.printf("SYNC_STOP v=2 t0=%lld t1=%lld on_us=%u gap_us=%u\n",
                (long long)t0, (long long)t1,
                (unsigned)led().timing().on_ms * 1000u,
                (unsigned)led().timing().gap_ms * 1000u);
}

// fields left unwritten keep the caller's default, so a short command is not zeros
static void splitFloats(const String &s, float *out, int n) {
  String rest = s;
  for (int i = 0; i < n && rest.length(); i++) {
    int c = rest.indexOf(':');
    out[i] = (c < 0 ? rest : rest.substring(0, c)).toFloat();
    rest = (c < 0) ? String() : rest.substring(c + 1);
  }
}

// indexed by the integer the host sends; same order as control/constants.py
static const TaskMode RAMP_MODES[] = {
    TaskMode::POLYNOMIAL, TaskMode::EASE, TaskMode::EXPONENTIAL};
static const int N_RAMP_MODES = sizeof(RAMP_MODES) / sizeof(RAMP_MODES[0]);

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
    const int64_t led0 = esp_timer_get_time();
    led().sendStartSignal();
    const int64_t led1 = esp_timer_get_time();
    const int64_t drive0 = esp_timer_get_time();
    seq.start();
    // first carrier application, before any blocking serial output
    applyMixer();
    const int64_t drive1 = esp_timer_get_time();
    state = SPINUP;
    Serial.printf("SYNC_START v=2 led0=%lld led1=%lld drive0=%lld drive1=%lld "
                  "on_us=%u gap_us=%u\n",
                  (long long)led0, (long long)led1,
                  (long long)drive0, (long long)drive1,
                  (unsigned)led().timing().on_ms * 1000u,
                  (unsigned)led().timing().gap_ms * 1000u);
    Serial.printf("seq go: %d segment(s) from %.2f Hz\n", seqN, seqF0);
    return true;
  }
  if (sub == "ramp") {
    float v[5] = {0, 0, 0, 0, 1.0f};
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

static constexpr uint8_t LED_WORD = 0b0110011;

// the LED word locates the LED, never a clock
static bool cmdLed(const String &arg) {
  if (arg != "code") { Serial.println("!led arg"); return false; }
  if (state != IDLE) { Serial.printf("!led state=%d\n", (int)state); return false; }
  led().sendCode(LED_WORD);
  Serial.printf("SYNC_CODE v=1 word=0x%02x bit_us=%u\n", LED_WORD,
                (unsigned)led().timing().code_ms * 1000u);
  return true;
}

// IDLE only: this energises, and the blocking burst services no button or stop
static const uint32_t PROBE_MAX_MS = 2000;
static const float PROBE_DUTY = 60.0f;
static const uint32_t PROBE_SETTLE_MS = 250;

static bool cmdProbe(const String &arg) {
  if (state != IDLE) { Serial.printf("!probe state=%d\n", (int)state); return false; }

  float v[3] = {0.0f, 1200.0f, PROBE_DUTY};
  splitFloats(arg, v, 3);
  if (!(v[0] > 0.0f)) { Serial.println("!probe needs hz>0"); return false; }
  const uint32_t ms = (uint32_t)clampf(v[1], 100.0f, (float)PROBE_MAX_MS);
  const float duty = clampf(v[2], 0.0f, 100.0f);

  ctl.setGlobalFrequency(v[0]);
  for (int i = 0; i < NUM_CHANNELS; i++) ctl.setCarrierDutyCycle(i, duty);
  delay(PROBE_SETTLE_MS);

  ProbeResult r;
  const bool ok = coilProbe(ADC_PINS, NUM_CHANNELS, v[0], ms, SPIN_PHASES, r);
  allCoilsOff();
  // back to DC idle, so the echoed frequency does not read as driving
  ctl.setGlobalFrequency(0.0f);

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
  else if (key == "stop")  { const bool wasDriven =
                               state == SPINUP || state == FLIGHT || state == LANDING;
                             allCoilsOff();
                             if (wasDriven) reportStop();
                             state = OFF; ok = true; }
  else if (key == "freq")   ok = cmdFreq(arg);
  else if (key == "seq")    ok = cmdSeq(arg);
  else if (key == "probe")  ok = cmdProbe(arg);
  else if (key == "duty")   ok = cmdDuty(arg);
  else if (key == "phase")  ok = cmdPhase(arg);
  else if (key == "led")    ok = cmdLed(arg);
  else {
    Serial.printf("? '%s' (seq=|throttle=|az=|mag=|duty=|phase=|led=code|hover|land|stop|freq=|probe=)\n",
                  cmd.c_str());
    return;
  }
  if (!ok) return;

  Serial.printf("state=%d col=%.0f az=%.0f mag=%.2f freq=%.2f duty=%s\n",
                (int)state, collective, azSet, magSet, ctl.getFrequency(),
                dutySet ? "set" : "off");
}

void setup() {
  driveBoot(false);
  ctl.begin();
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS,   0.0f);
  // trim stays off unless every table entry is positive
  ctl.setPhaseTrim(COIL_F0_HZ, COIL_Q);
  Serial.printf("[flight] phase trim %s\n",
                ctl.phaseTrimActive() ? "ARMED from drive_common.h" : "off (uncalibrated)");
  Serial.println("SYNC_READY v=1");
}

// amp = 0 in a frame is not a stop: only `stop` de-energises
static void applyDrive(const drive_frame::Drive &d) {
  // a frame must never drive the coils from IDLE
  if (state != SPINUP && state != FLIGHT) return;
  for (int i = 0; i < NUM_CHANNELS; i++) dutyPct[i] = clampf(d.amp_pct[i], 0.0f, 100.0f);
  dutySet = true;
  for (int i = 0; i < NUM_CHANNELS; i++) phaseDeg[i] = d.phase_deg[i];
  ctl.setPhases(phaseDeg);
  if (d.freq_hz > 0.0f) ctl.setGlobalFrequency(d.freq_hz);
  lastDriveSeq = d.seq;
  // no per-frame echo: the ACK carries the same at 10 Hz
  if (++sinceAck >= ACK_EVERY) {
    sinceAck = 0;
    drive_frame::Ack a;
    a.seq_echo = lastDriveSeq;
    a.n_rx = framer.n_rx();
    a.n_crc = (uint16_t)(framer.n_crc() + framer.n_gap());
    a.state = (uint8_t)state;
    uint8_t buf[drive_frame::ACK_LEN];
    drive_frame::encode_ack(a, buf);
    Serial.write(buf, drive_frame::ACK_LEN);
  }
}

void loop() {
  // drain everything waiting, or the 1 kB FIFO overflows silently
  while (Serial.available()) {
    drive_frame::Got g = framer.feed((uint8_t)Serial.read(), (uint32_t)micros());
    if (g == drive_frame::Got::Frame) applyDrive(framer.drive());
    else if (g == drive_frame::Got::Line) dispatch(String(framer.line()));
    else if (g == drive_frame::Got::Overflow)
      Serial.println("! line over 128 chars, truncated");
  }

  ctl.run();

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
        if (ctl.rampDownStep(2.0f)) {
          reportStop();
          state = OFF; Serial.println("state=4 (OFF)");
        }
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
