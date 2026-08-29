// Live PC-commanded flight: takeoff -> hover -> directional acceleration.
// Commands (newline, 115200): takeoff | throttle=<pct> | az=<deg> | mag=<0..1> |
// hover | land | stop | freq=<hz>. freq= is the altitude handle (controller/control/
// z_track.py), FLIGHT only, and arms a 500 ms silence watchdog that lands the robot if the
// host stops commanding. Current balancing is OFF, so setCarrierDutyCycle sets each
// channel's ceiling and run() balances thrust beneath it, so a differential
// ceiling tilts the disk. One flight per boot; reset to re-arm.
#include "drive_common.h"
#include "SerialComm.h"
#include "constants.h"

// Hardware knobs, tune on the rig.
static const float HOVER_HZ = 150.0f; // ramp target; NOT a measured hover point
static const unsigned long SPINUP_MS = 30000;
static const float SPINUP_THROTTLE = 100.0f;
// Physical coil azimuths A,B,C,D (deg). SEED GUESS: sweep az, see which pair weakens.
static const float COIL_AZ[NUM_CHANNELS] = {0.0f, 90.0f, 180.0f, 270.0f};
// Tilt authority: az-facing coils drop MIX_GAIN*mag. SEED GUESS, identify on the rig.
static const float MIX_GAIN = 0.6f;
// No frequency band here: every real limit lives on the PC, which owns the model. What is
// left is a corruption filter, since a truncated serial line parses to 0.0.
static const unsigned long CMD_TIMEOUT_MS = 500; // ~15 control periods at 30 Hz

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static PwmSequencer seq(&ctl);
static SerialComm comm;

enum State { IDLE, SPINUP, FLIGHT, LANDING, OFF };
static State state = IDLE;

static float collective = 0.0f; // % carrier ceiling (throttle)
static float azSet = 0.0f;      // deg
static float magSet = 0.0f;     // 0..1

// Command watchdog. Armed by the first mag=/freq= in FLIGHT, not on entry to it: the host
// sends nothing while it waits to be armed, and landing on that silence would be a fault.
static unsigned long lastCmdMs = 0;
static bool watchdogArmed = false;

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
    if (state == FLIGHT) { watchdogArmed = true; }
  } else if (cmd.startsWith("freq=")) {
    // FLIGHT only: seq.run() owns the frequency during SPINUP and would overwrite it.
    float hz = cmd.substring(5).toFloat();
    if (state != FLIGHT) { Serial.printf("!freq state=%d\n", (int)state); return; }
    // <=0 is DC: the field stops rotating, the robot drops, and coil current is limited
    // only by R. The PC clamps the flight band; this only catches a corrupt line.
    if (hz <= 0.0f) { Serial.printf("!freq=%.2f\n", hz); return; }
    watchdogArmed = true;
    ctl.setGlobalFrequency(hz);
  } else if (cmd == "hover") {
    magSet = 0.0f;
  } else if (cmd == "land") {
    if (state == SPINUP || state == FLIGHT) state = LANDING;
  } else if (cmd == "stop") {
    allCoilsOff(); state = OFF;
  } else {
    Serial.printf("? '%s' (takeoff|throttle=|az=|mag=|hover|land|stop|freq=)\n", cmd.c_str());
    return;
  }
  Serial.printf("state=%d col=%.0f az=%.0f mag=%.2f freq=%.2f\n",
                (int)state, collective, azSet, magSet, ctl.getFrequency());
}

void setup() {
  driveBoot();
  ctl.begin(); // DC; the ramp sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS, /*tripA*/ 10.0f);
  // ctl.enableCurrentBalance();  // OFF: CS is blind to the disk (tilt_ccw_nodisk), the
  // imbalance is static magnetic coupling that feedforward trims already fix (1.97 ->
  // 1.046), and the loop confounds the az sweep. Re-enable only with evidence.

  seq.addRampTask(1.0f, HOVER_HZ, SPINUP_MS, TaskType::PWM_FREQ, TaskMode::EASE);
  seq.compile(25, 1.0f, INITIAL_DUTY, PHASES_CCW);
  Serial.println("flight: IDLE -- send 'takeoff' to spin up");
}

void loop() {
  String line = comm.handleSerialComm();
  if (line.length()) { lastCmdMs = millis(); dispatch(line); }

  ctl.run(); // sense + balance + overcurrent trip

  switch (state) {
    case SPINUP:
      seq.run();
      applyMixer();
      if (seq.isDone()) { state = FLIGHT; Serial.println("state=2 (FLIGHT)"); }
      break;
    case FLIGHT:
      if (watchdogArmed && millis() - lastCmdMs > CMD_TIMEOUT_MS) {
        Serial.printf("!silence %lums, landing\n", millis() - lastCmdMs);
        state = LANDING;
        break;
      }
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
