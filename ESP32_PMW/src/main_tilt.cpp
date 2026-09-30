#include "drive_common.h"

// instantiate PWM controller and sequencer:
PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
JsonPwmSequencer seq(&ctl);

void setup() {
  driveBoot(); // from drive_common.h
  
  ctl.begin(); // DC (stationary); the schedule sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS);

  ctl.enableCurrentBalance(); // enable PI current balancing

  seq.loadFromJsonFile("/tilt.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();

  // experiment-specific behaviour: blink LED once per step
  static size_t lastStep = (size_t)-1;
  size_t step = seq.currentIndex(); // gets the current step in the task sequence
  if (step != lastStep) {
    lastStep = step;
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }

  // Announce the schedule's label whenever it changes. This firmware takes no
  // commands, so this line is the only event the host can put a clock on -- it is
  // what `controller/control/tilt_sweep.py` stamps and `sync.py` aligns to the
  // video. Printed on label change, not on step change: a 2.5 s hold compiles to
  // ~100 queue steps that all share one label.
  static String lastLabel = "\x01"; // not "" -- an unlabelled first step must print
  const String &label = seq.stepLabel();
  if (label != lastLabel) {
    lastLabel = label;
    Serial.printf("label=%s\n", label.c_str());
  }

  driveTelemetry(ctl);
}
