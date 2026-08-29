// Upside-down takeoff: CW phases, 100% carrier, EASE 1->100Hz, PI-balanced. The
// balance loop holds the channels together beneath the commanded carrier. Runs
// on boot -- no arming. Schedule (with its own setDirection=CW) lives in
// /takeoff_upside_down.json.
#include "drive_common.h"

static PwmController ctl(PWM_PINS, PHASES_CW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPwmSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin(); // DC (stationary); the schedule sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS);
  // ctl.enableCurrentBalance();  // OFF: CS is blind to the disk (tilt_ccw_nodisk), the
  // imbalance is static magnetic coupling that feedforward trims already fix (1.97 ->
  // 1.046), and the loop confounds the az sweep. Re-enable only with evidence.
  seq.loadFromJsonFile("/takeoff_upside_down.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();

  driveTelemetry(ctl);
}
