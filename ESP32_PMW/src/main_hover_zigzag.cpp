// Hover zigzag: ramp 1->160Hz then zigzag 160<->140Hz five times, 100% carrier,
// PI-balanced. Runs on boot -- no arming. Schedule (with its own
// setDirection=CCW) lives in /hover_zigzag.json.
#include "drive_common.h"

static PwmController ctl(PWM_PINS, PHASES_CW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPhaseSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin(); // DC (stationary); the schedule sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS);
  ctl.enableCurrentBalance();
  seq.loadFromJsonFile("/hover_zigzag.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();
  driveTelemetry(ctl); // also polls the block/restart button
}
