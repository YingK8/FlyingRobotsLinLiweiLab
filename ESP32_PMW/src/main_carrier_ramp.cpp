// Carrier-duty ramp: carrier 0->100% at fixed 190Hz, phases {0,180,90,270}. The
// PI balance loop holds the channels together beneath the ramping carrier
// ceiling. Schedule lives in /carrier_ramp.json.
#include "balanced_experiment.h"

void setup() {
  // jsonFile, ccwDefault, driveFreq, iSafetyMax, piEnabled
  experimentSetup(ExperimentConfig("/carrier_ramp.json", true, 190.0f, 10.0f, true));
}

void loop() { experimentLoop(); }
