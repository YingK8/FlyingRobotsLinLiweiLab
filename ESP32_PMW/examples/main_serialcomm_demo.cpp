// Minimal flashable demo/verification target for lib/SerialComm: echoes each
// incoming line back prefixed with "echo: ". Drive it with tools/serial_comm.py
// to confirm both directions work non-blocking over real hardware.
#include <Arduino.h>
#include "SerialComm.h"

SerialComm comm;

void setup() { Serial.begin(115200); }

void loop() {
  String line = comm.step();
  if (line.length()) {
    comm.step("echo: " + line);
  }
}
