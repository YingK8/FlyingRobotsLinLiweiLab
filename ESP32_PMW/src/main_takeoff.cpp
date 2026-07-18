// Takeoff: back-to-back EASE-then-LINEAR 1->500Hz ramps at 100% carrier, CCW,
// PI-balanced. The balance loop holds the channels together beneath the
// commanded carrier. Runs on boot -- no arming. Schedule lives in /takeoff.json.
#include "drive_common.h"

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPhaseSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin(); // DC (stationary); the schedule sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS);
  // ctl.enableCurrentBalance();
  seq.loadFromJsonFile("/takeoff.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();

  // --- experiment-specific behavior: LED steady-on while spinning up ---
  digitalWrite(LED_PIN, seq.isDone() ? LOW : HIGH);

  driveTelemetry(ctl);
}
