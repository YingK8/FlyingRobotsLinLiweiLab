#include "drive_common.h"

// instantiate PWM controller and sequencer:
PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
JsonPhaseSequencer seq(&ctl);

void setup() {
  driveBoot();
  
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

  driveTelemetry(ctl);
}
