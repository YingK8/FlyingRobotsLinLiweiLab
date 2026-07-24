// Live PC-commanded flight: takeoff -> hover -> directional acceleration.
// Commands (newline, 115200): takeoff | throttle=<pct> | az=<deg> | mag=<0..1> |
// hover | land | stop. With enableCurrentBalance on, setCarrierDutyCycle sets each
// channel's ceiling and run() balances thrust beneath it, so a differential
// ceiling tilts the disk. One flight per boot; reset to re-arm.
#include "drive_common.h"
#include "SerialComm.h"

// Hardware knobs, tune on the rig.
static const float HOVER_HZ = 150.0f; // ~130-157 Hz torque band
static const unsigned long SPINUP_MS = 30000;
static const float SPINUP_THROTTLE = 100.0f;
// Physical coil azimuths A,B,C,D (deg). SEED GUESS: sweep az, see which pair weakens.
static const float COIL_AZ[NUM_CHANNELS] = {0.0f, 90.0f, 180.0f, 270.0f};
// Tilt authority: az-facing coils drop MIX_GAIN*mag. Fit from ../writeup/single_results.csv.
static const float MIX_GAIN = 0.6f;

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static PhaseSequencer seq(&ctl);
static SerialComm comm;

enum State { IDLE, SPINUP, FLIGHT, LANDING, OFF };
static State state = IDLE;

static float collective = 0.0f; // % carrier ceiling (throttle)
static float azSet = 0.0f;      // deg
static float magSet = 0.0f;     // 0..1

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Thrust-vector mixer: drop the az-facing coils' ceilings so the disk tilts toward
// az. Strong side stays at collective, the balance reference. Verify sign on rig.
static void applyMixer() {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    float drop = MIX_GAIN * magSet * max(0.0f, cosf((azSet - COIL_AZ[i]) * (float)DEG_TO_RAD));
    ctl.setCarrierDutyCycle(i, clampf(collective * (1.0f - drop), 0.0f, 100.0f));
  }
}

static void allCoilsOff() {
  for (int i = 0; i < NUM_CHANNELS; i++) ctl.setCarrierDutyCycle(i, 0.0f);
}

static void dispatch(String cmd) {
  cmd.trim();
  cmd.toLowerCase();
  if (cmd == "takeoff") {
    if (state == IDLE) { collective = SPINUP_THROTTLE; seq.start(); state = SPINUP; }
  } else if (cmd.startsWith("throttle=")) {
    collective = clampf(cmd.substring(9).toFloat(), 0.0f, 100.0f);
  } else if (cmd.startsWith("az=")) {
    azSet = cmd.substring(3).toFloat();
  } else if (cmd.startsWith("mag=")) {
    magSet = clampf(cmd.substring(4).toFloat(), 0.0f, 1.0f);
  } else if (cmd == "hover") {
    magSet = 0.0f;
  } else if (cmd == "land") {
    if (state == SPINUP || state == FLIGHT) state = LANDING;
  } else if (cmd == "stop") {
    allCoilsOff(); state = OFF;
  } else {
    Serial.printf("? '%s' (takeoff|throttle=|az=|mag=|hover|land|stop)\n", cmd.c_str());
    return;
  }
  Serial.printf("state=%d col=%.0f az=%.0f mag=%.2f\n", (int)state, collective, azSet, magSet);
}

void setup() {
  driveBoot();
  ctl.begin(); // DC; the ramp sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS, /*tripA*/ 10.0f);
  ctl.enableCurrentBalance(); // PI holds the 4 currents beneath the mixed ceilings

  seq.addRampTask(1.0f, HOVER_HZ, SPINUP_MS, TaskType::PWM_FREQ, TaskMode::EASE);
  seq.compile(25, 1.0f, INITIAL_DUTY, PHASES_CCW);
  Serial.println("flight: IDLE -- send 'takeoff' to spin up");
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
