# CsvPhaseSequencer

Use CsvPhaseSequencer when you want to define PWM motion as time-based waypoints in a CSV file, then interpolate between those waypoints at a fixed step size.

This guide shows how to:
- Write a valid CSV waypoint file.
- Load it from SPIFFS.
- Run the sequencer in main.cpp.

## Prerequisites

- PhaseController and PhaseSequencer are already integrated in your project.
- SPIFFS is enabled and your CSV file is uploaded to the board filesystem.
- Your channels are indexed 0..3.

## CSV format

CsvPhaseSequencer accepts metadata lines (starting with #), then data lines.

### Metadata

- # channels, 4
- # step_size_ms, 25
- # interpolation, linear

Supported interpolation values:
- linear
- hermite
- ease

Notes:
- hermite and ease currently use the same smoothstep profile in the CSV sequencer.
- step_size_ms is the interpolation sample spacing for generated trajectory points.

### Data columns

Header:

```csv
time,channel,duty,carrier_duty,frequency,phase
```

Each row defines one channel value at one waypoint time.

Recommended pattern: for each time value, provide one row per channel (0, 1, 2, 3).

## How waypoints are interpolated

1. Rows with the same time are grouped into one waypoint.
2. Waypoint-to-waypoint segments are sampled every step_size_ms.
3. For each sample, duty, carrier duty, frequency, and phase are interpolated.
4. Generated samples are converted into TASK_TRAJECTORY_POINT sequence tasks.
5. If initialDuty or initialPhase is not provided, the first waypoint is used.

## Step-by-step: create a CSV profile

1. Add metadata lines for channels, step size, and interpolation mode.
2. Add the header row.
3. Add one row per channel for each waypoint time.
4. Keep channel index in range 0..3.

Example:

```csv
# channels, 4
# step_size_ms, 25
# interpolation, linear
time,channel,duty,carrier_duty,frequency,phase
0,0,50,100,10,0
0,1,50,100,10,90
0,2,50,100,10,180
0,3,50,100,10,270
200,0,60,100,12,0
200,1,60,100,12,90
200,2,40,80,12,180
200,3,40,80,12,270
```

## Step-by-step: use CsvPhaseSequencer in main.cpp

```cpp
#include "CsvPhaseSequencer.h"
#include "PhaseController.h"
#include <Arduino.h>
#include <SPIFFS.h>

const int NUM_CHANNELS = 4;
const gpio_num_t PWM_PINS[NUM_CHANNELS] = {
    GPIO_NUM_19, GPIO_NUM_33, GPIO_NUM_27, GPIO_NUM_32};
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0f, 90.0f, 180.0f, 270.0f};
const float INITIAL_DUTY[NUM_CHANNELS] = {50.0f, 50.0f, 50.0f, 50.0f};

PhaseController controller(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY, NUM_CHANNELS);
CsvPhaseSequencer seq(&controller);

void setup() {
  Serial.begin(115200);
  delay(300);

  if (!SPIFFS.begin(true)) {
    Serial.println("Failed to mount SPIFFS");
    return;
  }

  controller.begin(300.0f);

  if (!seq.loadFromCSVFile("/profile.csv", STEP_SIZE_MS)) {
    Serial.println("Failed to load CSV profile");
    return;
  }

  seq.start();
}

void loop() {
  controller.run();
  seq.run();
}
```

## Troubleshooting

- Failed to open CSV file:
  - Check SPIFFS mount and path (for example /profile.csv).
  - Make sure the file is uploaded to the board filesystem.

- Sequence looks wrong at first frame:
  - Confirm first waypoint has all 4 channels.
  - Pass explicit initialDuty and initialPhase, or ensure first waypoint matches desired startup state.

- Interpolation appears too coarse or too dense:
  - Tune # step_size_ms in the CSV metadata.
