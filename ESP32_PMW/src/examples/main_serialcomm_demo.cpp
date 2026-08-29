// Minimal flashable demo/verification target for lib/SerialComm: echoes each
// incoming line back prefixed with "echo: ". Drive it with `pio device monitor`
// to confirm both directions work non-blocking over real hardware.
#include <Arduino.h>
#include "SerialComm.h"

SerialComm comm;

void setup() { Serial.begin(115200); }

void loop() {
  String line = comm.handleSerialComm();
  if (line.length()) {
    comm.handleSerialComm("echo: " + line);
  }
}
