# SerialComm

Non-blocking, newline-framed serial send/receive for ESP32 sketches. Call
`handleSerialComm()` once per `loop()` iteration; it never blocks and never
depends on a whole line arriving inside one call.

Generalizes the reader that `main_tilt.cpp` had built inline. Framing is
plain ASCII terminated by `\n`, `\r`, or `\r\n` -- no checksum, no binary
framing -- matching the existing PC-side tools (`tools/run_experiment.py`,
`tools/trigger_reset_log.py`) and their Python mirror (`tools/serial_comm.py`).

## Usage

```cpp
#include "SerialComm.h"

SerialComm comm; // wraps Serial by default

void setup() {
  Serial.begin(115200);
}

void loop() {
  String line = comm.handleSerialComm(); // nothing to send this tick
  if (line.length()) {
    Serial.printf("got: %s\n", line.c_str());
  }

  // ...or send something while checking for incoming:
  // String line = comm.handleSerialComm("state=1 freq=7.2");
}
```

## Reference

```cpp
SerialComm(Stream &port = Serial);
String handleSerialComm(const String &outgoing = String());
```

`handleSerialComm` writes `outgoing` (if non-empty) plus a trailing `\n`,
drains whatever bytes are currently available, and returns the first
complete line that finished this call, or `""` if none did. If multiple
lines are waiting, only one is returned per call -- the rest stay buffered
for the next call, so nothing is dropped.
