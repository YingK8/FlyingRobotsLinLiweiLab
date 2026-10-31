#include "drive_common.h"

// no software stop: this env starts on reset and parses no serial
static PwmController ctl(PWM_PINS, PHASES_CW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPwmSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin();
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS);
  // balance stays off: the current sense is blind to the disk
  seq.loadFromJsonFile("/takeoff_upside_down.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();

  driveTelemetry(ctl);
}
