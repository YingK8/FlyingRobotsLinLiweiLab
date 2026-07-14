// Takeoff experiment: back-to-back EASE-then-LINEAR 1->500Hz/400s ramps at 100%
// carrier, phases {0,180,90,270}. The PI balance loop holds the channels
// together beneath the commanded carrier. Schedule lives in /takeoff.json.
#include "balanced_experiment.h"

void setup() {
  // jsonFile, ccwDefault, driveFreq, iSafetyMax, piEnabled
  experimentSetup(ExperimentConfig("/takeoff.json", true, 190.0f, 10.0f, true));
}

void loop() { experimentLoop(); }
