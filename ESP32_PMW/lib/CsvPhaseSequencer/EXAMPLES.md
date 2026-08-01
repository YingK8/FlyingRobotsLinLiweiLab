# CsvPhaseSequencer CSV Examples

All examples define waypoints. CsvPhaseSequencer interpolates between waypoints
using # step_size_ms and # interpolation.

## Example 1: Minimal hold profile

This creates one waypoint and holds it.

```csv
# channels, 4
# step_size_ms, 25
# interpolation, linear
time,channel,duty,carrier_duty,frequency,phase
0,0,50,100,10,0
0,1,50,100,10,90
0,2,50,100,10,180
0,3,50,100,10,270
```

## Example 2: Linear interpolation between two waypoints

```csv
# channels, 4
# step_size_ms, 20
# interpolation, linear
time,channel,duty,carrier_duty,frequency,phase
0,0,50,100,10,0
0,1,50,100,10,90
0,2,50,100,10,180
0,3,50,100,10,270
200,0,80,100,15,0
200,1,80,100,15,90
200,2,20,60,15,180
200,3,20,60,15,270
```

## Example 3: Smooth easing interpolation over multiple waypoints

Use hermite (or ease) to smooth transitions.

```csv
# channels, 4
# step_size_ms, 25
# interpolation, hermite
time,channel,duty,carrier_duty,frequency,phase
0,0,40,100,8,0
0,1,40,100,8,90
0,2,40,100,8,180
0,3,40,100,8,270
300,0,65,100,12,0
300,1,65,100,12,90
300,2,35,70,12,180
300,3,35,70,12,270
700,0,55,90,10,0
700,1,55,90,10,90
700,2,45,90,10,180
700,3,45,90,10,270
```

## Example 4: File + load call pairing

CSV file path in SPIFFS:

```text
/profile.csv
```

Loader call in setup:

```cpp
if (!seq.loadFromCSVFile("/profile.csv", STEP_SIZE_MS)) {
  Serial.println("Failed to load CSV profile");
}
```

## Common mistakes

Missing channel rows for a waypoint time:

```csv
# Wrong: time 200 has only channel 0
200,0,70,100,12,0
```

Correct:

```csv
200,0,70,100,12,0
200,1,70,100,12,90
200,2,30,80,12,180
200,3,30,80,12,270
```

Unknown interpolation value:

```csv
# Wrong
# interpolation, cubic
```

Correct:

```csv
# interpolation, linear
```
