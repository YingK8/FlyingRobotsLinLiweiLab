# ExperimentPhase

Shared header-only primitives for the `ARMING -> WAITING -> <running> ->
STOPPED/DONE` state machine duplicated (near-verbatim) across
`main_experiment.cpp`, `main_current_pid.cpp`, and `main_pi_profile.cpp`.

Firmware never auto-starts off a timer: `ARMING` always transitions to
`WAITING` (coils latched off) and stays there until an explicit start command
arrives over serial (`isStartCommand()`). An e-stop command
(`isStopCommand()`) is checked first, unconditionally, in every phase.

## Reference

```cpp
bool armingTick(unsigned long now, unsigned long phaseStart, unsigned long armMs,
                 PWMController *controller, int numChannels,
                 CurrentSense &currentSense, int ledPin);
```
Coils off, fast-blink LED, `currentSense.recalibrateZero()` (valid because
coils are confirmed off). Returns `true` once `armMs` has elapsed since
`phaseStart` -- the caller should transition to `WAITING`, not directly to a
running phase.

```cpp
void latchedOffTick(unsigned long now, PWMController *controller,
                     int numChannels, int ledPin);
```
Coils off, slow heartbeat LED. Used by every parked/terminal phase
(`WAITING`, `STOPPED`, `DONE`, `*_FAILED`).

```cpp
bool isStopCommand(const String &cmd);   // "s" / "stop" / "estop"
bool isStartCommand(const String &cmd);  // "r" / "run" / "start" / "go"
```
`cmd` should already be trimmed and lowercased (same convention as each
file's existing `dispatchCommand()`).
