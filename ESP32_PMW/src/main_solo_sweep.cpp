// Per-channel solo frequency sweep: one coil energized at a time, stepped and
// held across 100..220 Hz so a handheld gaussmeter can be read by hand at each
// point. PASSTHROUGH (no enableCurrentBalance) -- balancing would erase the
// per-channel difference this measures. Schedule lives in /solo_sweep.json
// (safely latches off at the end). Prints the step label on change so the
// operator knows which channel/frequency is live while reading the meter.
#include "drive_common.h"

static PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY, NUM_CHANNELS);
static JsonPhaseSequencer seq(&ctl);

void setup() {
  driveBoot();
  ctl.begin(); // DC (stationary); the schedule sets the running frequency
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  ctl.enableCurrentSense(ADC_PINS, SENS, /*tripA*/ 10.0f); // no balance: passthrough
  seq.loadFromJsonFile("/solo_sweep.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();

  // --- experiment-specific behavior: announce each step, and TOGGLE the LED on
  // every step edge. Keep the LED in frame when filming the gaussmeter: each
  // toggle marks a step boundary, which is how the video aligns to this log.
  static size_t lastStep = (size_t)-1;
  size_t step = seq.currentIndex();
  if (step != lastStep) {
    lastStep = step;
    Serial.printf(">>> STEP %u: %s\n", (unsigned)step, seq.labelForStep(step));
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
  if (seq.isDone())
    digitalWrite(LED_PIN, LOW);

  driveTelemetry(ctl);
}
