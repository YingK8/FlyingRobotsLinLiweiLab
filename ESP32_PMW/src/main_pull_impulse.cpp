// Pulsed-gradient conveyor: release the robot from one coil, kick it toward the
// next, coast, then hold a weak guide field. The source coil is fully off
// during the kick, so it cannot suck the robot back.
// All coils are driven at the 50 Hz series resonance with a 50 % commutation
// duty; the sequence modulates carrier amplitudes only. Label order is
// A -> D -> B -> C -> A (the firmware's clockwise order).
// PASSTHROUGH (no enableCurrentBalance): the deliberate imbalance is the
// experiment.
// Runs on boot -- no arming. Schedule lives in /pull_impulse.json.
#include "drive_common.h"

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPwmSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin(); // safe idle; seq.start() applies the JSON's initial_freq
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS, /*tripA*/ 10.0f); // no balance: passthrough
  seq.loadFromJsonFile("/pull_impulse.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();
  driveTelemetry(ctl);
}
