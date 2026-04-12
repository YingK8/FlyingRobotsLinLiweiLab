#include "CsvPhaseSequencer.h"

namespace {

float hermite01(float t) {
  if (t < 0.0f) {
    t = 0.0f;
  } else if (t > 1.0f) {
    t = 1.0f;
  }
  return t * t * (3.0f - 2.0f * t);
}

float interpolateValue(float startValue, float endValue, float alpha,
                       bool useHermite) {
  if (useHermite) {
    return startValue + (endValue - startValue) * hermite01(alpha);
  }
  return startValue + (endValue - startValue) * alpha;
}

SequenceTask makeTaskFromPoint(const TrajectoryPoint &point, int numChannels,
                               int64_t durationUs) {
  SequenceTask task = {};
  task.type = TASK_TRAJECTORY_POINT;
  task.durationUs = durationUs;
  task.startFreq = point.freq[0];
  task.endFreq = point.freq[0];
  for (int channel = 0; channel < numChannels; ++channel) {
    task.dutyCycles[channel] = point.duty[channel];
    task.carrierDuties[channel] = point.carrierDuties[channel];
    task.startPhases[channel] = point.phase[channel];
    task.endPhases[channel] = point.phase[channel];
  }
  return task;
}

} // namespace

CsvPhaseSequencer::CsvPhaseSequencer(PhaseController *phaseCtrl)
    : PhaseSequencer(phaseCtrl) {}

bool CsvPhaseSequencer::loadFromCSVFile(const char *filename,
                                        uint32_t resolutionMs) {
  File file = SPIFFS.open(filename, "r");
  if (!file) {
    Serial.println("[CsvPhaseSequencer] Failed to open CSV file");
    return false;
  }

  int numChannels = 4;
  uint32_t stepSizeMs = (resolutionMs > 0) ? resolutionMs : STEP_SIZE_MS;
  String interpolationType = "linear";

  std::vector<TrajectoryPoint> csvRows;
  String header;
  bool headerFound = false;
  int64_t currentTimeMs = -1;
  TrajectoryPoint currentPoint = {};
  bool hasCurrentPoint = false;

  while (file.available()) {
    String line = file.readStringUntil('\n');
    line.trim();

    if (line.length() == 0) {
      continue;
    }

    if (line.startsWith("#")) {
      if (line.startsWith("# channels")) {
        int idx = line.indexOf(',');
        if (idx > 0) {
          numChannels = line.substring(idx + 1).toInt();
        }
      } else if (line.startsWith("# step_size_ms")) {
        int idx = line.indexOf(',');
        if (idx > 0) {
          stepSizeMs = static_cast<uint32_t>(line.substring(idx + 1).toInt());
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

    int lastIdx = 0;
    int nextIdx = 0;
    int64_t timeMs = 0;
    int channel = 0;
    float valDuty = 0.0f;
    float valCarrier = 0.0f;
    float valFreq = 0.0f;
    float valPhase = 0.0f;

    for (int column = 0; column < 6; ++column) {
      nextIdx = line.indexOf(',', lastIdx);
      String token = (nextIdx == -1) ? line.substring(lastIdx)
                                     : line.substring(lastIdx, nextIdx);
      token.trim();

      switch (column) {
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

      if (nextIdx == -1) {
        break;
      }
      lastIdx = nextIdx + 1;
    }

    if (timeMs != currentTimeMs) {
      if (hasCurrentPoint) {
        csvRows.push_back(currentPoint);
      }
      currentPoint = {};
      currentPoint.timeUs = timeMs * 1000LL;
      currentTimeMs = timeMs;
      hasCurrentPoint = true;
    }

    if (channel >= 0 && channel < numChannels) {
      currentPoint.duty[channel] = valDuty;
      currentPoint.carrierDuties[channel] = valCarrier;
      currentPoint.freq[channel] = valFreq;
      currentPoint.phase[channel] = valPhase;
    }
  }

  if (hasCurrentPoint) {
    csvRows.push_back(currentPoint);
  }

  file.close();

  if (csvRows.empty()) {
    Serial.println("[CsvPhaseSequencer] CSV file contained no trajectory rows");
    return false;
  }

  if (csvRows.front().timeUs != 0) {
    Serial.println(
        "[CsvPhaseSequencer] First CSV block must start at time 0 ms");
    return false;
  }

  bool useHermite =
      interpolationType == "hermite" || interpolationType == "ease";

  std::vector<SequenceTask> generatedTasks;
  generatedTasks.reserve(csvRows.size() * 4);

  if (csvRows.size() == 1) {
    generatedTasks.push_back(
        makeTaskFromPoint(csvRows.front(), numChannels,
                          static_cast<int64_t>(stepSizeMs) * 1000LL));
  } else {
    for (size_t segment = 0; segment + 1 < csvRows.size(); ++segment) {
      const TrajectoryPoint &startPoint = csvRows[segment];
      const TrajectoryPoint &endPoint = csvRows[segment + 1];
      int64_t segmentDurationUs = endPoint.timeUs - startPoint.timeUs;
      if (segmentDurationUs <= 0) {
        continue;
      }

      int64_t sampleStepUs = static_cast<int64_t>(stepSizeMs) * 1000LL;
      int64_t elapsedUs = 0;
      while (elapsedUs < segmentDurationUs) {
        float alpha = static_cast<float>(elapsedUs) /
                      static_cast<float>(segmentDurationUs);
        TrajectoryPoint sample = {};
        sample.timeUs = startPoint.timeUs + elapsedUs;

        for (int channel = 0; channel < numChannels; ++channel) {
          sample.duty[channel] =
              interpolateValue(startPoint.duty[channel], endPoint.duty[channel],
                               alpha, useHermite);
          sample.carrierDuties[channel] = interpolateValue(
              startPoint.carrierDuties[channel],
              endPoint.carrierDuties[channel], alpha, useHermite);
          sample.freq[channel] =
              interpolateValue(startPoint.freq[channel], endPoint.freq[channel],
                               alpha, useHermite);
          sample.phase[channel] =
              interpolateValue(startPoint.phase[channel],
                               endPoint.phase[channel], alpha, useHermite);
        }

        int64_t remainingUs = segmentDurationUs - elapsedUs;
        int64_t durationUs =
            remainingUs < sampleStepUs ? remainingUs : sampleStepUs;
        if (durationUs <= 0) {
          durationUs = sampleStepUs;
        }

        generatedTasks.push_back(
            makeTaskFromPoint(sample, numChannels, durationUs));
        elapsedUs += durationUs;
      }
    }

    generatedTasks.push_back(
        makeTaskFromPoint(csvRows.back(), numChannels,
                          static_cast<int64_t>(stepSizeMs) * 1000LL));
  }

  for (const SequenceTask &task : generatedTasks) {
    addSequenceTask(task);
  }

  float csvInitialDuty[4] = {0, 0, 0, 0};
  float csvInitialPhase[4] = {0, 0, 0, 0};
  for (int channel = 0; channel < numChannels; ++channel) {
    csvInitialDuty[channel] = csvRows.front().duty[channel];
    csvInitialPhase[channel] = csvRows.front().phase[channel];
  }
  const float csvInitialFreq = csvRows.front().freq[0];

  compile(stepSizeMs, csvInitialFreq, csvInitialDuty, csvInitialPhase);
  return true;
}