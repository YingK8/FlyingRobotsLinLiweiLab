// Carrier-duty ramp: carrier 0->100% at a fixed 190Hz, CCW, PI-balanced. The
// balance loop holds the channels together beneath the ramping carrier ceiling.
// Runs on boot -- no arming. Schedule lives in /carrier_ramp.json.
#include "drive_common.h"

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPhaseSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin(); // DC (stationary); the schedule sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS, /*tripA*/ 10.0f);
  ctl.enableCurrentBalance();
  seq.loadFromJsonFile("/carrier_ramp.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();

  // --- experiment-specific behavior: blink the LED once per schedule step ---
  static size_t lastStep = (size_t)-1;
  size_t step = seq.currentIndex();
  if (step != lastStep) {
    lastStep = step;
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }

  driveTelemetry(ctl);
}
