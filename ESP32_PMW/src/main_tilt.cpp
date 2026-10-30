#include "drive_common.h"

// instantiate PWM controller and sequencer:
PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
JsonPwmSequencer seq(&ctl);

void setup() {
  driveBoot(false);
  
  ctl.begin(); // DC (stationary); the schedule sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS);

  // NO enableCurrentBalance() -- deliberately PASSTHROUGH, as the calibration rigs in
  // src/calibration/ are and for the same reason. Measured 2026-09-01, 150 Hz hold with
  // channel A commanded to 20% carrier:
  //
  //   duty[%]: A=20.0 B=46.6 C=58.0 D=47.5   I[A]: A=-1.38 B=2.45 C=2.43 D=2.44
  //
  // A's sense reads NEGATIVE, so it is permanently the balancer's ratio-normalised
  // argmin, and the loop drags the other three down to ~47-58% chasing it. The commanded
  // 20:100:100:100 field became 20:47:58:48 -- the asymmetry was set by the fault, not by
  // the sweep. Closing this loop on a bad measurement erases the effect being measured.
  // See `controller/control/theory.md` 23.1.

  seq.loadFromJsonFile("/tilt.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();

  static String lastLabel = "\x01"; // not "" -- an unlabelled first step must print
  const String &label = seq.stepLabel();
  if (label != lastLabel) {
    lastLabel = label;
    led().sendBlink();
    Serial.printf("label=%s\n", label.c_str());
  }

  driveTelemetry(ctl);
}
