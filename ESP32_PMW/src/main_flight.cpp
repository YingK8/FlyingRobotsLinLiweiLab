// Live PC-commanded flight: takeoff -> hover -> directional acceleration.
// Commands (newline, 115200): takeoff | throttle=<pct> | az=<deg> | mag=<0..1> |
// hover | land | stop | freq=<hz>. freq= is the altitude-loop handle (ai/z_track.py),
// accepted in FLIGHT only. With enableCurrentBalance on, setCarrierDutyCycle sets each
// channel's ceiling and run() balances thrust beneath it, so a differential
// ceiling tilts the disk. One flight per boot; reset to re-arm.
#include "drive_common.h"
#include "SerialComm.h"
#include "constants.h"

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static PwmSequencer seq(&ctl);
static SerialComm comm;

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
  } else if (cmd.startsWith("freq=")) {
    // FLIGHT only: seq.run() owns the frequency during SPINUP and would overwrite
    // this on its next tick. Reject out-of-band rather than clamp, so a corrupt
    // line is visible in the log instead of silently flying at the limit.
    float hz = cmd.substring(5).toFloat();
    if (state != FLIGHT) { Serial.printf("!freq state=%d\n", (int)state); return; }
    if (hz < FREQ_MIN || hz > FREQ_MAX) { Serial.printf("!freq=%.2f\n", hz); return; }
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
