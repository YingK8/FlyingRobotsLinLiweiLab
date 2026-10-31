#include "drive_common.h"

// no software stop: this env starts on reset and parses no serial
static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY,
                                 NUM_CHANNELS);
static JsonPwmSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin();
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  // no balance: the capture wants the schedule's duty held as commanded
  ctl.enableCurrentSense(ADC_PINS, SENS, 10.0f);
  seq.loadFromJsonFile("/dc_calibration.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();

  if (seq.isDone()) led().holdOff(); else led().holdOn();

  driveTelemetry(ctl);
}
