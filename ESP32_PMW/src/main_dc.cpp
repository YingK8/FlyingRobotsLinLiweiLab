// DC calibration: 100% commutation duty + 100% carrier on all channels parks
// every pin statically HIGH through the normal driver path, for a DC current-
// sense calibration capture. Passthrough (piEnabled=false): no balancing.
// Schedule lives in /dc_calibration.json (safely latches off after its window).
#include "balanced_experiment.h"

void setup() {
  // jsonFile, ccwDefault, driveFreq, iSafetyMax, piEnabled
  experimentSetup(ExperimentConfig("/dc_calibration.json", true, 190.0f, 10.0f, false));
}

void loop() { experimentLoop(); }
