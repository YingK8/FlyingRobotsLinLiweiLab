// Swim setup; set SWIM_SETUP=1 in platformio.ini to enable this code (should be the default when upload wth env:swim)
#include "drive_common.h"

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPhaseSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin(); // DC (stationary); the schedule sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  seq.loadFromJsonFile("/swim.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();
  driveTelemetry(ctl); // also polls the block/restart button
}
