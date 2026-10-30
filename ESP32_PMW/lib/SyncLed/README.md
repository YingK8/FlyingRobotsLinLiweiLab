# SyncLed

The one writer of `LED_PIN`, and the optical marker a recording is aligned to.

## Setup

```cpp
#include "SyncLedPin.h"

void setup() { led().attachTo(LED_PIN); }   // after forceAllGatesLow(); starts dark
void loop()  { led().advance(); }          // at most one edge, never blocks
```

`driveBoot()` already does both calls for you.

## Use

| call | edges | ends |
|---|---|---|
| `sendStartSignal()` | ON, OFF after `on_ms`, ON after `gap_ms` | ON |
| `sendStopSignal()` | OFF, ON after `on_ms`, OFF after `gap_ms` | OFF |
| `sendBlink()` | ON, OFF after `blink_ms` | OFF |
| `holdOn()` / `holdOff()` | one edge, if the level changes | — |

The first edge is the anchor; the second confirms it. Both must be seen or the host
refuses the take.

## Rules

- **Call `advance()` every loop**, or the second edge never happens.
- **Never `delay()` around a signal.** `sendStartSignal()` writes its first edge
  synchronously so `esp_timer_get_time()` can bracket it; the rest rides on `advance()`.
- **30 ms is a duration, not a frame count.** It is ≥3 frame periods at 120–210 fps and
  not a multiple of either, so it cannot be missed or beat. Change it in `PulseTiming`.
- **A signal whose first edge changes nothing is not armed** — a stuck-on LED emits
  nothing rather than a phantom falling edge.
- `holdOn()`/`holdOff()` cancel a running signal. An interrupted signal never resumes.

## Self-check

```bash
c++ -std=c++17 -O2 -I lib/SyncLed/src -o /tmp/tsl lib/SyncLed/test_sync_led.cpp && /tmp/tsl
```

Why two edges and why not a Hamming code: `controller/control/theory.md`, "Recorded
flight START". Host side: `LLM/START_SYNC.md`.
