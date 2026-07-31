// Swim: spin up to a swim frequency, then undulate around it to produce a
// stroke. PI-balanced, runs on boot -- no arming. The motion itself (and its
// own direction=CCW) lives in /swim.json; regenerate it with the generator in
// ai/ (see README) rather than editing it by hand.
#include "drive_common.h"

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPhaseSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin(); // DC (stationary); the schedule sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS);
  ctl.enableCurrentBalance();
  seq.loadFromJsonFile("/swim.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();
  driveTelemetry(ctl); // also polls the block/restart button
}
