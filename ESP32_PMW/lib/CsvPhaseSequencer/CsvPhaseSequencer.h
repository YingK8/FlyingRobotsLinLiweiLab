
#pragma once
#include "PhaseController.h"
#include "PhaseSequencer.h"
#include <Arduino.h>
#include <FS.h>
#include <SPIFFS.h>
#include <vector>

const uint32_t STEP_SIZE_MS = 25; // Default time step for trajectory points

class CsvPhaseSequencer : public PhaseSequencer {
public:
  CsvPhaseSequencer(PhaseController *phaseCtrl);

  /**
   * @brief Load waypoints from a CSV file (SPIFFS) and compile the sequence.
   *
   * Header row is `time,channel,duty,carrier_duty,frequency,phase`; optional
   * `# channels,N` / `# step_size_ms,N` / `# interpolation,linear|hermite`
   * directives precede it. Rows sharing a `time` form one full-state point;
   * points are resampled every `resolutionMs` and interpolated between (the
   * `# interpolation` directive picks linear vs. hermite/ease).
   *
   * @return False if the file can't be opened, has no rows, or its first
   *   block doesn't start at time 0.
   */
  bool loadFromCSVFile(const char *filename,
                       uint32_t resolutionMs = STEP_SIZE_MS);
};
