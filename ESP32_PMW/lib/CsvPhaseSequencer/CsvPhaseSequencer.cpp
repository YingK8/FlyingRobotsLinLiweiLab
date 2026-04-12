
#include "CsvPhaseSequencer.h"
#include <FS.h>

CsvPhaseSequencer::CsvPhaseSequencer(PhaseController *phaseCtrl)
    : PhaseSequencer(phaseCtrl) {}

// Skeleton for CsvPhaseSequencer methods

bool CsvPhaseSequencer::loadFromCSVFile(const char *filename,
                                        uint32_t resolutionMs,
                                        float initialFreq,
                                        const float *initialDuty,
                                        const float *initialPhase) {
  // Open the CSV file (SPIFFS or SD)
  File file = SPIFFS.open(filename, "r");
  if (!file) {
    Serial.println("[CsvPhaseSequencer] Failed to open CSV file");
    return false;
  }

  // --- Metadata defaults ---
  int numChannels = 4;
  uint32_t stepSizeMs = 25;
  String interpolationType = "linear";

  // --- Parse lines ---
  String header;
  bool headerFound = false;
  while (file.available()) {
    String line = file.readStringUntil('\n');
    line.trim();
    if (line.length() == 0)
      continue;
    if (line.startsWith("#")) {
      // Metadata line
      if (line.startsWith("# channels")) {
        int idx = line.indexOf(',');
        if (idx > 0) {
          numChannels = line.substring(idx + 1).toInt();
        }
      } else if (line.startsWith("# step_size_ms")) {
        int idx = line.indexOf(',');
        if (idx > 0) {
          stepSizeMs = line.substring(idx + 1).toInt();
        }
      } else if (line.startsWith("# interpolation")) {
        int idx = line.indexOf(',');
        if (idx > 0) {
          interpolationType = line.substring(idx + 1);
          interpolationType.trim();
        }
      }
      continue;
    }
    if (!headerFound) {
      header = line;
      headerFound = true;
      continue;
    }

    // Data row
    // Columns: time,channel,duty,carrier_duty,frequency,phase
    int col = 0;
    int lastIdx = 0;
    int nextIdx = 0;
    float duty[4] = {0}, carrierDuty[4] = {0}, freq[4] = {0}, phase[4] = {0};
    int64_t timeMs = 0;
    int channel = 0;
    float valDuty = 0, valCarrier = 0, valFreq = 0, valPhase = 0;

    SequenceTask prev_task;
    float prev_

        // Parse CSV columns
        for (int i = 0; i < 6; ++i) {
      nextIdx = line.indexOf(',', lastIdx);
      String token = (nextIdx == -1) ? line.substring(lastIdx)
                                     : line.substring(lastIdx, nextIdx);
      token.trim();
      switch (i) {
      case 0:
        timeMs = token.toInt();
        break;
      case 1:
        channel = token.toInt();
        break;
      case 2:
        valDuty = token.toFloat();
        break;
      case 3:
        valCarrier = token.toFloat();
        break;
      case 4:
        valFreq = token.toFloat();
        break;
      case 5:
        valPhase = token.toFloat();
        break;
      }
      if (nextIdx == -1)
        break;
      lastIdx = nextIdx + 1;
    }

    // Assign values to arrays for the channel
    if (channel >= 0 && channel < numChannels) {
      duty[channel] = valDuty;
      carrierDuty[channel] = valCarrier;
      freq[channel] = valFreq;
      phase[channel] = valPhase;
    }

    // Store the parsed row for later interpolation
    static std::vector<SequenceTask> csvRows;
    static std::vector<int64_t> csvTimes;
    if (csvRows.size() <= 0 || csvTimes.size() <= 0 ||
        timeMs != csvTimes.back()) {
      SequenceTask task;
      task.type = TASK_TRAJECTORY_POINT;
      task.durationUs = timeMs * 1000; // ms to us
      for (int c = 0; c < numChannels; ++c) {
        task.dutyCycles[c] = duty[c];
        task.carrierDuties[c] = carrierDuty[c];
        task.startFreq = freq[c];
        task.endFreq = freq[c];
        task.startPhases[c] = phase[c];
        task.endPhases[c] = phase[c];
      }
      csvRows.push_back(task);
      csvTimes.push_back(timeMs);
    }
  }

  file.close();

  // Interpolate between time steps
  // Use csvRows and csvTimes to generate interpolated SequenceTasks
  if (csvRows.size() < 2) {
    Serial.println("[CsvPhaseSequencer] Not enough data for interpolation");
    return false;
  }

  for (size_t seg = 0; seg < csvRows.size() - 1; ++seg) {
    int64_t t0 = csvTimes[seg];
    int64_t t1 = csvTimes[seg + 1];
    SequenceTask &row0 = csvRows[seg];
    SequenceTask &row1 = csvRows[seg + 1];
    int steps = (t1 - t0) / stepSizeMs;
    if ((t1 - t0) % stepSizeMs != 0)
      ++steps;
    for (int s = 0; s < steps; ++s) {
      float alpha = (steps == 1) ? 1.0f : (float)s / (float)steps;
      SequenceTask interpTask;
      interpTask.type = TASK_TRAJECTORY_POINT;
      interpTask.durationUs = (t0 + s * stepSizeMs) * 1000;
      for (int c = 0; c < numChannels; ++c) {
        float y0 = row0.dutyCycles[c];
        float y1 = row1.dutyCycles[c];
        float p0 = row0.startPhases[c];
        float p1 = row1.startPhases[c];
        float v0 = 0, v1 = 0;
        // Linear interpolation
        if (interpolationType == "linear") {
          interpTask.dutyCycles[c] = y0 + (y1 - y0) * alpha;
          interpTask.carrierDuties[c] =
              row0.carrierDuties[c] +
              (row1.carrierDuties[c] - row0.carrierDuties[c]) * alpha;
          interpTask.startPhases[c] = p0 + (p1 - p0) * alpha;
          interpTask.endPhases[c] = interpTask.startPhases[c];
        } else if (interpolationType == "hermite" ||
                   interpolationType == "ease") {
          // Hermite (ease) interpolation with C1 continuity
          // For endpoints, use finite difference for tangent
          if (seg == 0)
            v0 = (y1 - y0) / (t1 - t0);
          else
            v0 = (row1.dutyCycles[c] - csvRows[seg - 1].dutyCycles[c]) /
                 (csvTimes[seg + 1] - csvTimes[seg - 1]);
          if (seg == csvRows.size() - 2)
            v1 = (y1 - y0) / (t1 - t0);
          else
            v1 = (csvRows[seg + 2].dutyCycles[c] - y0) /
                 (csvTimes[seg + 2] - t0);
          float t = alpha;
          float h00 = 2 * t * t * t - 3 * t * t + 1;
          float h10 = t * t * t - 2 * t * t + t;
          float h01 = -2 * t * t * t + 3 * t * t;
          float h11 = t * t * t - t * t;
          interpTask.dutyCycles[c] =
              h00 * y0 + h10 * (t1 - t0) * v0 + h01 * y1 + h11 * (t1 - t0) * v1;
          // For phase, use linear for now
          interpTask.carrierDuties[c] =
              row0.carrierDuties[c] +
              (row1.carrierDuties[c] - row0.carrierDuties[c]) * alpha;
          interpTask.startPhases[c] = p0 + (p1 - p0) * alpha;
          interpTask.endPhases[c] = interpTask.startPhases[c];
        }
        interpTask.startFreq =
            row0.startFreq + (row1.startFreq - row0.startFreq) * alpha;
        interpTask.endFreq = interpTask.startFreq;
      }
      addSequenceTask(interpTask);
    }
  }

  // Set initial values from first row if not provided
  float defaultDuty[4] = {50, 50, 50, 50};
  float defaultPhase[4] = {0, 90, 180, 270};
  if (!initialDuty) {
    for (int i = 0; i < numChannels; ++i) {
      defaultDuty[i] = csvRows[0].dutyCycles[i];
    }
    initialDuty = defaultDuty;
  }
  if (!initialPhase) {
    for (int i = 0; i < numChannels; ++i) {
      defaultPhase[i] = csvRows[0].startPhases[i];
    }
    initialPhase = defaultPhase;
  }
  compile(stepSizeMs, initialFreq, initialDuty, initialPhase);
  return true;
}

// ...other methods skeletons...
