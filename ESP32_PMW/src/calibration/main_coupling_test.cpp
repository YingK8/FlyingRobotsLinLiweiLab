#include "drive_common.h"

// no software stop: this env starts on reset and parses no serial
static PwmController ctl(PWM_PINS, PHASES_CW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPwmSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin();
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  // no balance: it would equalise the channels this sweep drives unequally
  ctl.enableCurrentSense(ADC_PINS, SENS, 10.0f);
  seq.loadFromJsonFile("/coupling_cw.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();

  static size_t lastStep = (size_t)-1; // -1 so task 0 still blinks
  size_t step = seq.currentIndex();
  if (step != lastStep) {
    lastStep = step;
    led().flip();
  }

  driveTelemetry(ctl);
}
