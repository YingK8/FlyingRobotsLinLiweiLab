// DC calibration: 100% commutation duty + 100% carrier on all channels parks
// every pin statically HIGH through the normal driver path, for a DC current-
// sense calibration capture. PASSTHROUGH (no enableCurrentBalance): no balancing.
// Runs on boot -- no arming. Schedule lives in /dc_calibration.json (safely
// latches off after its window).
#include "drive_common.h"

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPhaseSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin(); // DC (stationary); the schedule sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS, /*tripA*/ 10.0f); // no balance: passthrough
  seq.loadFromJsonFile("/dc_calibration.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();

  // --- experiment-specific behavior: LED steady-on while the DC window holds ---
  digitalWrite(LED_PIN, seq.isDone() ? LOW : HIGH);

  driveTelemetry(ctl);
}
