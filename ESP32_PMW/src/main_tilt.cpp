// Tilt experiment: EASE 1->210Hz frequency ramp, then a 100%->0% carrier
// step-down (10% steps, 2.5s holds), CCW. The step-down commands one overall
// ceiling per step; the shared PI current-balance loop holds the four channels
// together (< 0.4 A spread) beneath it -- so the per-channel trims are found
// automatically instead of hand-tuned. Schedule lives in /tilt.json.
#include "balanced_experiment.h"

void setup() {
  // jsonFile, ccwDefault, driveFreq, iSafetyMax, piEnabled
  experimentSetup(ExperimentConfig("/tilt.json", true, 190.0f, 10.0f, true));
}

void loop() { experimentLoop(); }
