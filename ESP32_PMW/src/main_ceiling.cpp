// Ceiling characterization run: all four channels held at 100% carrier while the
// drive frequency EASE-ramps 1->210Hz, open-loop (piEnabled=false). Gives each
// channel's peak unregulated current vs frequency, Ceiling_i(f), which
// tools/tilt_metrics.py divides the regulated currents by to get utilization.
// Schedule lives in /ceiling_sweep.json.
#include "balanced_experiment.h"

void setup() {
  // jsonFile, ccwDefault, driveFreq, iSafetyMax, piEnabled
  experimentSetup(ExperimentConfig("/ceiling_sweep.json", true, 190.0f, 10.0f, false));
}

void loop() { experimentLoop(); }
