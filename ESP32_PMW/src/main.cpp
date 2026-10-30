// JSON-driven horizontal rotating field.
// All behavior lives in /horizontal_10hz.json:
//   10 Hz, 50 % commutation duty, phases A/B/C/D = 0/90/180/90,
//   equal 50 % carrier on all four coils.
// The payload sets the phases and carrier; the sequencer applies them at start
// and the field continues after the (single-task) schedule completes.
#include "drive_common.h"

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPwmSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin(); // safe idle; seq.start() applies the JSON's initial_freq
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  // ctl.enableCurrentSense(ADC_PINS, SENS, /*tripA*/ 10.0f);
  seq.loadFromJsonFile("/horizontal_10hz.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();
  driveTelemetry(ctl);
}
