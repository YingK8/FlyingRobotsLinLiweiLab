# SerialComm

Non-blocking, newline-framed serial send/receive for ESP32 sketches. Call
`handleSerialComm()` once per `loop()` iteration; it never blocks and never
depends on a whole line arriving inside one call.

Generalizes a reader that the experiment mains used to build inline. Framing is
plain ASCII terminated by `\n`, `\r`, or `\r\n`, mirrored on the host by
`controller/control/link.py`.

**Optional checksum.** A line ending in `*hh` carries CRC-8/SMBUS (poly 0x07, init 0)
of everything before the `*`. A bad one prints `!crc <line>` and is dropped. A line
without `*hh` is accepted as-is, so typing into `pio device monitor` still works; the
price is that corruption which deletes the `*` gets through. The host appends it only
after `hello` answers `crc=1`, and never on `stop`/`land`, so the kill parses on any
firmware.

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
