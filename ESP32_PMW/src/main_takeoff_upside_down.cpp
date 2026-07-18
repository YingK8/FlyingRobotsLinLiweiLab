// Upside-down takeoff: CW phases, 100% carrier, EASE 1->190Hz, PI-balanced. The
// balance loop holds the channels together beneath the commanded carrier. Runs
// on boot -- no arming. Schedule (with its own setDirection=CW) lives in
// /takeoff_upside_down.json.
#include "drive_common.h"

static PwmController ctl(PWM_PINS, PHASES_CW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPhaseSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin(); // DC (stationary); the schedule sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS, /*tripA*/ 10.0f);
  ctl.enableCurrentBalance();
  seq.loadFromJsonFile("/takeoff_upside_down.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();

  // --- experiment-specific behavior: LED steady-on while spinning up ---
  digitalWrite(LED_PIN, seq.isDone() ? LOW : HIGH);

  driveTelemetry(ctl);
}
