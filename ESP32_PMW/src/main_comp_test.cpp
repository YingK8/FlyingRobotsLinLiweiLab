// Compensation A/B test: BASELINE (equal 50% duty) -> GAP -> TRIMMED
// (per-channel trim), CCW. PASSTHROUGH (no enableCurrentBalance): this measures
// the effect of *specific* per-channel trims, so the balance loop must NOT
// override them -- current sense stays on only for telemetry + the overcurrent
// latch. Runs on boot -- no arming. Schedule lives in /comp_test.json.
#include "drive_common.h"

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPhaseSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin(); // DC (stationary); the schedule sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS, /*tripA*/ 10.0f); // no balance: passthrough
  seq.loadFromJsonFile("/comp_test.json");
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
