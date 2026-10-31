#include "drive_common.h"

// no software stop: this env starts on reset and parses no serial
PwmController ctl(PWM_PINS, PHASES_CCW, INITIAL_DUTY,
                          NUM_CHANNELS);
JsonPwmSequencer seq(&ctl);

void setup() {
  driveBoot(false);

  ctl.begin();
  ctl.initCarrierPWM(CARRIER_PINS, PWM_FREQ, CARRIER_ZERO);
  // balance stays off: a bad sense reading sets the asymmetry this sweep measures
  ctl.enableCurrentSense(ADC_PINS, SENS);

  seq.loadFromJsonFile("/tilt.json");
  seq.start();
}

void loop() {
  seq.run();
  ctl.run();

  static String lastLabel = "\x01"; // not empty: an unlabelled first task must print
  const String &label = seq.stepLabel();
  if (label != lastLabel) {
    lastLabel = label;
    led().sendBlink();
    Serial.printf("label=%s\n", label.c_str());
  }

  driveTelemetry(ctl);
}
