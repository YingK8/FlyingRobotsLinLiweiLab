// Upside-down takeoff: CW phases, 100% carrier, EASE 1->190Hz/40s. The PI
// balance loop holds the channels together beneath the commanded carrier.
// Schedule (with its own setDirection=CW) lives in /takeoff_upside_down.json.
#include "balanced_experiment.h"

void setup() {
  // jsonFile, ccwDefault, driveFreq, iSafetyMax, piEnabled
  experimentSetup(
      ExperimentConfig("/takeoff_upside_down.json", false, 190.0f, 10.0f, true));
}

void loop() { experimentLoop(); }
