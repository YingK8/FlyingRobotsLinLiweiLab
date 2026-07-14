// Compensation test: BASELINE (equal 50% duty) -> GAP -> TRIMMED (per-channel
// trim) A/B comparison, CCW. Passthrough (piEnabled=false): this measures the
// effect of *specific* per-channel trims, so the balance loop must NOT override
// them. Schedule lives in /comp_test.json.
#include "balanced_experiment.h"

void setup() {
  // jsonFile, ccwDefault, driveFreq, iSafetyMax, piEnabled
  experimentSetup(ExperimentConfig("/comp_test.json", true, 190.0f, 10.0f, false));
}

void loop() { experimentLoop(); }
